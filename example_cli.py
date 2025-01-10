#!/usr/bin/env python

import os
import argparse
import sys
import json
from os.path import exists, dirname, basename, join, expanduser
from glob import glob
import pandas as pd
import numpy as np


HELP_PARAGRAPHS = {
    'main':'an example cli tool',
    'level1': {
        'main':'main level',
        'first-thing':'docs for first thing',
        'second-thing':'docs for second thing',
    },
    'level2': {
        'main':'level 2',
        'third-thing':'docs for third thing',
        'fourth-thing':'docs for fourth thing',
    },
}


def check_path(fp):
    # Check if the path is absolute or relative
    if os.path.isabs(args.input_file):
        file_path = args.input_file
    else:
        file_path = os.path.join(os.getcwd(), args.input_file)

    # Verify if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")

    return file_path


def main():
    parser = argparse.ArgumentParser(description=HELP_PARAGRAPHS['main'])
    subparsers = parser.add_subparsers(dest='command', help='')

    h = HELP_PARAGRAPHS['level1']
    parser_corpus = subparsers.add_parser('level1', help=h['main'])
    parser_corpus.add_argument('--first-thing', '-f', required=False,
                               default='', action='store',
                               help=h['first-thing'])
    parser_corpus.add_argument('--second-thing', '-s', required=False,
                               default=False, action='store_true',
                               help=h['second-thing'])

    h = HELP_PARAGRAPHS['level2']
    parser_summ = subparsers.add_parser('level2', help=h['main'])
    parser_summ.add_argument('--third-thing', '-t', required=False,
                             default=False, action='store_true',
                             help=h['third-thing'])
    parser_summ.add_argument('--fourth-thing', '-f', required=False,
                             default=False, action='store_true',
                             help=h['fourth-thing'])
    args = parser.parse_args()

    if args.command == 'level1':
        if args.first_thing:
            print('here is first thing')
        elif args.second_thing:
            print('here is second thing')

    elif args.command == 'level2':
        if args.third_thing:
            print('3rd')
        elif args.fourth_thing:
            print('4th')

if __name__ == '__main__':
    main()
