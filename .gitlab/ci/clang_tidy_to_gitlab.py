import sys
import json
import re
import argparse


def parse_clang_tidy_output(input_file):
    """
    Parses clang-tidy output from a file and converts it to a list of dictionaries
    in the GitLab Code Quality format.

    Args:
        input_file (str): The path to the file containing clang-tidy output.

    Returns:
        list: A list of dictionaries, where each dictionary represents a single
              code quality issue in the GitLab format.  Returns an empty list
              if no issues are found or if there's an error.
    """
    issues = []
    try:
        with open(input_file, "r") as f:
            for line in f:
                # Example clang-tidy output line:
                # path/to/file.cpp:10:5: warning: ... [check_name]
                match = re.match(r"^(.*?):(\d+):(\d+): (\w+): (.*) \[(.*)\]", line)
                if match:
                    filepath, line_num, col_num, severity, message, check_name = (
                        match.groups()
                    )
                    # Convert line_num to int
                    line_num = int(line_num)
                    # Map clang-tidy severity to GitLab severity.  Crucial for
                    # proper display in GitLab.  clang-tidy uses:
                    #   error, warning, note
                    # GitLab Code Quality uses:
                    #   blocker, critical, major, minor, info
                    if severity == "error":
                        gl_severity = "critical"  #  Or "blocker" -  adjust as needed
                    elif severity == "warning":
                        gl_severity = "major"  # Or "minor" - adjust as needed
                    elif severity == "note":
                        gl_severity = "info"
                    else:
                        gl_severity = "info"  # Default, or handle unknown.

                    issue = {
                        "description": message,
                        "check_name": check_name,
                        "severity": gl_severity,
                        "location": {
                            "path": filepath,
                            "lines": {
                                "begin": line_num,
                                "end": line_num,  #  clang-tidy doesn't give end line.
                            },
                        },
                        "fingerprint": f"{filepath}:{line_num}:{message}",
                    }
                    issues.append(issue)
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_file}")
        return []
    except Exception as e:
        print(f"Error processing file: {e}")
        return []
    return issues


def write_gitlab_code_quality_report(issues, output_file):
    """
    Writes the list of code quality issues to a JSON file in the GitLab
    Code Quality report format.

    Args:
        issues (list): A list of dictionaries, where each dictionary
                        represents a code quality issue.
        output_file (str): The path to the output JSON file.
    """
    try:
        with open(output_file, "w") as f:
            json.dump(issues, f, indent=2)
        print(f"Successfully wrote GitLab Code Quality report to {output_file}")
    except Exception as e:
        print(f"Error writing output file: {e}")


def main():
    """
    Main function to parse clang-tidy output and generate a GitLab Code Quality report.
    Now handles multiple input files.
    """
    parser = argparse.ArgumentParser(
        description="Convert clang-tidy output to GitLab Code Quality report format."
    )
    parser.add_argument(
        "input_files", nargs="+", help="Path(s) to the clang-tidy output file(s)."
    )
    parser.add_argument(
        "output_file", help="Path to the output GitLab Code Quality JSON file."
    )

    args = parser.parse_args()
    input_files = args.input_files  # Now a list
    output_file = args.output_file

    all_issues = []
    for input_file in input_files:
        issues = parse_clang_tidy_output(input_file)
        if issues:
            all_issues.extend(issues)  # Extend the list, don't append the list.

    if all_issues:
        write_gitlab_code_quality_report(all_issues, output_file)
    else:
        print("No clang-tidy issues found, or error occurred.  No report generated.")
        sys.exit(0)


if __name__ == "__main__":
    main()
