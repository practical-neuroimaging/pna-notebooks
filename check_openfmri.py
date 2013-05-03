import sys
from openfmri import get_subjects
from checkhash import check_store_hash

def main():
    hash_dict = {}
    subjects = get_subjects(sys.argv[1])
    for name, subject in subjects.items():
        for run in subject['functionals']:
            check_store_hash(run['filename'], hash_dict)


if __name__ == '__main__':
    main()
