#!/usr/bin/env python3
import subprocess
import sys

def get_files_from_last_n_commits(n):
    result = subprocess.run(
        ['git', 'log', f'-{n}', '--name-only', '--pretty=format:'],
        capture_output=True,
        text=True,
        check=True
    )

    files = set()
    for line in result.stdout.strip().split('\n'):
        line = line.strip()
        if line:
            files.add(line)

    return sorted(files)

def main():
    if len(sys.argv) != 2:
        print("Usage: python get_commit_files.py <number_of_commits>")
        sys.exit(1)

    try:
        n_commits = int(sys.argv[1])
    except ValueError:
        print("Error: argument must be an integer")
        sys.exit(1)

    files = get_files_from_last_n_commits(n_commits)

    output_file = 'commit_files.txt'
    with open(output_file, 'w') as f:
        for file_path in files:
            f.write(f'{file_path}\n')

    print(f"Wrote {len(files)} file paths to {output_file}")

if __name__ == '__main__':
    main()
