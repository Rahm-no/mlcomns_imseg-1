import sys
import re

def count_cases(log1, log2):
    
    cases_1 = {}
    p_start = re.compile(r'^Epoch (\d+)')
    p_case = re.compile(r'(\d+)')

    epoch = 0
    with open(log1, 'r') as infile:
        for line in infile:

            if match := re.match(p_start, line):
                epoch = int(match.group(1))
                if epoch not in cases_1:
                    cases_1[epoch] = set()
            
            if match := re.match(p_case, line):
                case = match.group(1)
                cases_1[epoch].add(case)

    cases_2 = {}

    epoch = 0
    with open(log2, 'r') as infile:
        for line in infile:

            if match := re.match(p_start, line):
                epoch = int(match.group(1))
                if epoch not in cases_2:
                    cases_2[epoch] = set()
            
            if match := re.match(p_case, line):
                case = match.group(1)
                cases_2[epoch].add(case)

    for (epoch, cases1), (_, cases2) in zip(cases_1.items(), cases_2.items()):
        common_cases = cases1.intersection(cases2)
        print(f'Epoch {epoch}\nFound {len(common_cases)} cases in common: {common_cases}\n')


if __name__ == '__main__':
    count_cases(sys.argv[1], sys.argv[2])