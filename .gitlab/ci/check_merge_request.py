# This script is run after a successful merge request pipeline.
# It compares the runtime of all RTL simulation jobs on the master branch with the runtime of the same jobs on the merge request branch.
# A comment describing the runtime difference is added to the merge request.
# If the pipeline has a smaller or equal runtime, the merge request is approved.

import os
import requests
import pandas as pd
import re
import tempfile
import subprocess

headers = {
    "PRIVATE-TOKEN": os.environ["MERGE_REQUEST_ACCESS_TOKEN"],
}
base_url = f"https://code.stanford.edu/api/v4/projects/{os.environ['CI_PROJECT_ID']}/"


def get_latest_pipeline_on_branch(branch):
    response = requests.get(
        base_url + "pipelines",
        headers=headers,
        params={"ref": branch, "per_page": 1},
    ).json()

    return response[0]["id"]


def get_job_artifacts(job_id, tmpdir, path):
    """Download the artifacts for a job and return if the specified path exists."""
    response = requests.get(
        base_url + f"jobs/{job_id}/artifacts",
        headers=headers,
    )

    if response.status_code == 404:
        return False
    else:
        with open(os.path.join(tmpdir, "artifacts.zip"), "wb") as f:
            f.write(response.content)
        subprocess.check_output(
            ["unzip", os.path.join(tmpdir, "artifacts.zip")], cwd=tmpdir
        )

        return os.path.exists(os.path.join(tmpdir, *path))


def get_regression_dataframe(job_id):
    regression_results_path = ["regression_results", "latest", "test_results.pkl"]

    with tempfile.TemporaryDirectory() as tmpdir:
        success = get_job_artifacts(job_id, tmpdir, regression_results_path)
        if not success:
            return None
        else:
            return pd.read_pickle(os.path.join(tmpdir, *regression_results_path))


def get_pipeline_results(pipeline_id):
    response = requests.get(
        base_url + f"pipelines/{pipeline_id}/jobs",
        headers=headers,
        params={"per_page": 100},
    ).json()

    df = pd.DataFrame()

    for job in response:
        # only check RTL Simulation jobs
        if job["name"].startswith("RTL Simulation"):
            # job name has pattern "RTL Simulation (datatype, rowsxcols)"
            # but some jobs can also be named "RTL Simulation (datatype, ROWSxCOLS, INPUT_BUFFER_SIZExWEIGHT_BUFFER_SIZExOUTPUT_BUFFER_SIZE)"
            pattern = r"RTL Simulation \((?P<datatype>\w+), (?P<rows>\d+)x(?P<cols>\d+)(?:, (?P<input_buffer>\d+)x(?P<weight_buffer>\d+)x(?P<output_buffer>\d+))?\)"
            match = re.match(pattern, job["name"])

            if match:
                # extract datatype, rows, cols, input_buffer, weight_buffer, output_buffer
                datatype = match.group("datatype")
                rows = int(match.group("rows"))
                cols = int(match.group("cols"))
                input_buffer = match.group("input_buffer")
                weight_buffer = match.group("weight_buffer")
                output_buffer = match.group("output_buffer")
                job_df = get_regression_dataframe(job["id"])
                if job_df is None:
                    continue

                job_df["datatype"] = datatype
                job_df["rows"] = rows
                job_df["cols"] = cols
                job_df["input_buffer_size"] = (
                    int(input_buffer) if input_buffer else 1024
                )
                job_df["weight_buffer_size"] = (
                    int(weight_buffer) if weight_buffer else 1024
                )
                job_df["output_buffer_size"] = (
                    int(output_buffer) if output_buffer else 1024
                )
                df = pd.concat([df, job_df])

    return df


def main():
    # get latest pipeline result on master branch
    master_pipeline_id = get_latest_pipeline_on_branch("master")
    master_df = get_pipeline_results(master_pipeline_id)

    # get latest pipeline result on merge request branch
    merge_request_pipeline_id = os.environ["CI_PIPELINE_ID"]
    merge_request_df = get_pipeline_results(merge_request_pipeline_id)

    # for both dataframes, group by datatype, rows, cols, input_buffer_size, weight_buffer_size, output_buffer_size, and Model, and sum the runtime
    master_grouped = master_df.groupby(
        [
            "datatype",
            "rows",
            "cols",
            "input_buffer_size",
            "weight_buffer_size",
            "output_buffer_size",
            "Model",
        ]
    )["Runtime"].sum()

    merge_request_grouped = merge_request_df.groupby(
        [
            "datatype",
            "rows",
            "cols",
            "input_buffer_size",
            "weight_buffer_size",
            "output_buffer_size",
            "Model",
        ]
    )["Runtime"].sum()

    # merge the two dataframes
    merged = pd.merge(
        master_grouped,
        merge_request_grouped,
        left_index=True,
        right_index=True,
        suffixes=("_master", "_merge_request"),
        how="outer",
    )

    # calculate the runtime difference
    merged["Runtime_diff"] = merged["Runtime_master"] - merged["Runtime_merge_request"]

    # find where the runtime difference is negative
    negative_diff = merged[merged["Runtime_diff"] < 0]

    # note down configurations where the runtime difference is negative
    comment = ""
    if not negative_diff.empty:
        comment += "⚠️⚠️ The following configurations have a longer runtime on the merge request branch:\n\n"
        comment += "```\n"
        comment += negative_diff.to_string(index=False) + "\n"
        comment += "```\n"
        approve = False
    else:
        comment += "✅ The runtime of all configurations is the same or shorter on the merge request branch.\n"
        approve = True

    print(comment)

    # add the comment to the merge request
    response = requests.post(
        base_url + f"merge_requests/{os.environ['CI_MERGE_REQUEST_IID']}/notes",
        headers=headers,
        json={"body": comment},
    )

    if approve:
        requests.post(
            base_url + f"merge_requests/{os.environ['CI_MERGE_REQUEST_IID']}/approve",
            headers=headers,
        )
    else:
        requests.post(
            base_url + f"merge_requests/{os.environ['CI_MERGE_REQUEST_IID']}/unapprove",
            headers=headers,
        )


if __name__ == "__main__":
    main()
