import csv
import functools
import logging
import os
import random
from collections import Counter
from dataclasses import dataclass
import re

from speechbrain.dataio.dataio import (
    load_pkl,
    merge_csvs,
    read_audio_info,
    save_pkl,
)
from speechbrain.utils.data_utils import get_all_files
from speechbrain.utils.parallel import parallel_map

logger = logging.getLogger(__name__)
OPT_FILE = "opt_data_prepare.pkl"
SAMPLERATE = 16000
OPEN_SLR_11_NGRAM_MODELs = [
    "lm_tgsmall.arpa.gz",
    "lm_tglarge_timit.arpa.gz",
    "lm_tglarge.arpa.gz",
    "lm_tglarge_LIBRI.arpa.gz",
]

def prepare_librispeech(
    data_folder,
    save_folder,
    tr_splits=[],
    dev_splits=[],
    te_splits=[],
    select_n_sentences=None,
    merge_lst=[],
    merge_name=None,
    create_lexicon=False,
    skip_prep=False,
    lexicon_path=None,  # New parameter for local lexicon
):
    """
    Prepares csv files for a custom dataset with modifications from LibriSpeech script.

    Arguments are similar to original prepare_librispeech function.
    Added lexicon_path for local lexicon file.
    """
    if skip_prep:
        return
    
    splits = tr_splits + dev_splits + te_splits
    save_folder = save_folder
    select_n_sentences = select_n_sentences
    conf = {
        "select_n_sentences": select_n_sentences,
    }

    # Saving folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_opt = os.path.join(save_folder, OPT_FILE)

    # Check if this phase is already done (if so, skip it)
    if skip(splits, save_folder, conf):
        logger.info("Skipping preparation, completed in previous run.")
        return
    else:
        logger.info("Data_preparation...")

    # Additional checks to make sure the data folder contains dataset
    check_dataset_folders(data_folder, splits)

    # create csv files for each split
    all_texts = {}
    for split_index in range(len(splits)):
        split = splits[split_index]

        # Change .flac to .wav
        wav_lst = get_all_files(
            os.path.join(data_folder, split), match_and=[".wav"]
        )

        # Look for text file in the split folder
        text_file_path = os.path.join(data_folder, split, "text")
        if not os.path.isfile(text_file_path):
            raise FileNotFoundError(f"Text file not found in {text_file_path}")

        text_dict = text_to_dict(text_file_path)
        all_texts.update(text_dict)

        if select_n_sentences is not None:
            n_sentences = select_n_sentences[split_index]
        else:
            n_sentences = len(wav_lst)

        create_csv(save_folder, wav_lst, text_dict, split, n_sentences)

    # Merging csv file if needed
    if merge_lst and merge_name is not None:
        merge_files = [split_data + ".csv" for split_data in merge_lst]
        merge_csvs(
            data_folder=save_folder, csv_lst=merge_files, merged_csv=merge_name
        )

    # Create lexicon.csv and oov.csv
    if create_lexicon:
        create_lexicon_and_oov_csv(all_texts, save_folder, lexicon_path)

    # saving options
    save_pkl(conf, save_opt)


def create_lexicon_and_oov_csv(all_texts, save_folder, lexicon_path=None):
    """
    Modified to use local lexicon file.
    """
    # Get list of all words in the transcripts
    transcript_words = Counter()
    for key in all_texts:
        # Process transcript words: lowercase, remove punctuation
        processed_text = re.sub(r'[^\w\s]', '', all_texts[key].lower())
        transcript_words.update(processed_text.split())

    # Get list of all words in the lexicon
    lexicon_words = []
    lexicon_pronunciations = []
    with open(lexicon_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            word = line.split()[0].lower()
            pronunciation = line.split()[1:]
            lexicon_words.append(word)
            lexicon_pronunciations.append(pronunciation)

    # Create lexicon.csv
    header = "ID,duration,char,phn\n"
    lexicon_csv_path = os.path.join(save_folder, "lexicon.csv")
    with open(lexicon_csv_path, "w") as f:
        f.write(header)
        for idx in range(len(lexicon_words)):
            separated_graphemes = [c for c in lexicon_words[idx]]
            duration = len(separated_graphemes)
            graphemes = " ".join(separated_graphemes)
            pronunciation_no_numbers = [
                p.strip("0123456789") for p in lexicon_pronunciations[idx]
            ]
            phonemes = " ".join(pronunciation_no_numbers)
            line = (
                ",".join([str(idx), str(duration), graphemes, phonemes]) + "\n"
            )
            f.write(line)
    logger.info("Lexicon written to %s." % lexicon_csv_path)

    # Split lexicon.csv in train, validation, and test splits
    split_lexicon(save_folder, [98, 1, 1])


def download_openslr_librispeech_lm(destination, rescoring_lm=True):
    """
    Loads language models locally instead of downloading.

    Arguments
    ---------
    destination : str
        Directory where the language models are located.
    rescoring_lm : bool
        Flag for using the larger rescoring LM (not applicable for local).

    """
    os.makedirs(destination, exist_ok=True)
    for lm_file in OPEN_SLR_11_NGRAM_MODELs:
        lm_path = os.path.join(destination, lm_file)
        if not os.path.exists(lm_path):
            raise FileNotFoundError(f"Language model file {lm_file} not found in {destination}.")
    logger.info("Language models successfully loaded from %s." % destination)


def process_line(wav_file, text_dict) -> 'LSRow':
    # Change file extension from .flac to .wav
    snt_id = os.path.splitext(os.path.basename(wav_file))[0]
    
    # Extract speaker ID from file path (assuming structure: data_folder/split/spk_id/utterance.wav)
    spk_id = os.path.basename(os.path.dirname(wav_file))
    
    # Process text: lowercase, remove punctuation
    wrds = text_dict.get(snt_id, '')
    wrds = re.sub(r'[^\w\s]', '', wrds.lower())

    info = read_audio_info(wav_file)
    duration = info.num_frames / info.sample_rate

    return LSRow(
        snt_id=snt_id,
        spk_id=spk_id,
        duration=duration,
        file_path=wav_file,
        words=wrds,
    )


def text_to_dict(text_file_path):
    """
    Converts text file to dictionary.

    Assumes text file format: utterance_id text_content
    """
    text_dict = {}
    with open(text_file_path, "r") as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                text_dict[parts[0]] = parts[1]
    return text_dict
  
def download_librispeech_vocab_text(destination):
    """Download librispeech vocab file and unpack it."""
    local_vocab_path = "/path/to/local/librispeech-vocab.txt"
    # Copy the file instead of downloading
    import shutil
    shutil.copy(local_vocab_path, destination)


def check_dataset_folders(data_folder, splits):
    """
    Check if the data folder contains the required dataset structure
    """
    for split in splits:
        split_folder = os.path.join(data_folder, split)
        if not os.path.exists(split_folder):
            raise OSError(f"The folder {split_folder} does not exist")


@dataclass
class LSRow:
    snt_id: str
    spk_id: str
    duration: float
    file_path: str
    words: str


def skip(splits, save_folder, conf):
    """
    Detect when the data prep can be skipped.
    """
    # Checking csv files
    skip = True

    for split in splits:
        if not os.path.isfile(os.path.join(save_folder, split + ".csv")):
            skip = False

    #  Checking saved options
    save_opt = os.path.join(save_folder, OPT_FILE)
    if skip is True:
        if os.path.isfile(save_opt):
            opts_old = load_pkl(save_opt)
            if opts_old == conf:
                skip = True
            else:
                skip = False
        else:
            skip = False

    return skip


def create_csv(save_folder, wav_lst, text_dict, split, select_n_sentences):
    """
    Create the dataset csv file given a list of wav files.
    """
    # Setting path for the csv file
    csv_file = os.path.join(save_folder, split + ".csv")
    if os.path.exists(csv_file):
        logger.info("Csv file %s already exists, not recreating." % csv_file)
        return

    # Preliminary prints
    msg = "Creating csv lists in  %s..." % (csv_file)
    logger.info(msg)

    csv_lines = [["ID", "duration", "wav", "spk_id", "wrd"]]

    snt_cnt = 0
    line_processor = functools.partial(process_line, text_dict=text_dict)
    # Processing all the wav files in wav_lst
    # FLAC metadata reading is already fast, so we set a high chunk size
    # to limit main thread CPU bottlenecks
    for row in parallel_map(line_processor, wav_lst, chunk_size=8192):
        csv_line = [
            row.snt_id,
            str(row.duration),
            row.file_path,
            row.spk_id,
            row.words,
        ]

        # Appending current file to the csv_lines list
        csv_lines.append(csv_line)

        snt_cnt = snt_cnt + 1

        # parallel_map guarantees element ordering so we're OK
        if snt_cnt == select_n_sentences:
            break

    # Writing the csv_lines
    with open(csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_lines:
            csv_writer.writerow(line)

    # Final print
    msg = "%s successfully created!" % (csv_file)
    logger.info(msg)


def split_lexicon(data_folder, split_ratio):
    """
    Splits the lexicon.csv file into train, validation, and test csv files
    """
    # Reading lexicon.csv
    lexicon_csv_path = os.path.join(data_folder, "lexicon.csv")
    with open(lexicon_csv_path, "r") as f:
        lexicon_lines = f.readlines()
    # Remove header
    lexicon_lines = lexicon_lines[1:]

    # Shuffle entries
    random.shuffle(lexicon_lines)

    # Selecting lines
    header = "ID,duration,char,phn\n"

    tr_snts = int(0.01 * split_ratio[0] * len(lexicon_lines))
    train_lines = [header] + lexicon_lines[0:tr_snts]
    valid_snts = int(0.01 * split_ratio[1] * len(lexicon_lines))
    valid_lines = [header] + lexicon_lines[tr_snts : tr_snts + valid_snts]
    test_lines = [header] + lexicon_lines[tr_snts + valid_snts :]

    # Saving files
    with open(os.path.join(data_folder, "lexicon_tr.csv"), "w") as f:
        f.writelines(train_lines)
    with open(os.path.join(data_folder, "lexicon_dev.csv"), "w") as f:
        f.writelines(valid_lines)
    with open(os.path.join(data_folder, "lexicon_test.csv"), "w") as f:
        f.writelines(test_lines)

