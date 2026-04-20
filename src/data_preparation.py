import os
import json

from sklearn.model_selection import train_test_split

import pandas as pd

from utlities import develop_argument_map_from_corpus, develop_data_files, count_tokens

if __name__ == "__main__":

    # path to folder containing the different IAT-annotated corpora as downloaded from AIFdb.
    input_dirs = "Raw JSON Argument Maps (IAT)"
    # temporary path to save IAT - BAS converted files
    repurposed_json_folder = "/content/content/repurposed_maps"
    # final path to contain the uniform and cleaned data files.
    output_dir_for_corpus = "final"

    os.makedirs(repurposed_json_folder, exist_ok=True)
    os.makedirs(output_dir_for_corpus, exist_ok=True)

    # For collecting all dataset statistics
    all_stats = []

    for input_dir in os.listdir(input_dirs):
        input_dir = os.path.join(input_dirs, input_dir)

        # repurpose the argument maps

        develop_argument_map_from_corpus(input_dir, repurposed_json_folder)

        # add conversation text, do initial cleaning, filter out short graphs, develop the final csv and context examples.
        folder_name = str(os.path.basename(input_dir))

        # develop the new data files, and context examples
        df, context_examples, stats = develop_data_files(input_dir, repurposed_json_folder)
        # attach dataset name to stats and record it
        stats["dataset_name"] = folder_name
        all_stats.append(stats)

        # save the data files
        final_data = os.path.join(output_dir_for_corpus, "final_data")
        os.makedirs(final_data, exist_ok=True)
        df.to_csv(os.path.join(final_data, f"{folder_name}.csv"), index=False)

        if "qt" in folder_name.lower():
            training_set, testing_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42,
                )
            training_set.reset_index(drop=True, inplace=True)
            testing_set.reset_index(drop=True, inplace=True)

            training_set.to_csv(os.path.join(final_data, "QT30_training.csv"))
            testing_set.to_csv(os.path.join(final_data,"QT30_test.csv"))

        # save the context examples
        complex_examples_path = os.path.join(output_dir_for_corpus, "context_examples")
        os.makedirs(complex_examples_path, exist_ok=True)
        context_path = os.path.join(complex_examples_path, f"{folder_name}_context_examples.json")
        with open(context_path,  "w") as f:
            json.dump(context_examples, f, indent=2)

        for i, ctx in enumerate(context_examples):

          print(f"Context example {i+1} (ID: {ctx['conversation_id']}) has {count_tokens(ctx['conversation_text'])} tokens.")

        print(f"Saved {len(df)} rows to CSV and {len(context_examples)} context examples to JSON.")

    # Convert all collected stats into a summary dataframe
    stats_df = pd.DataFrame(all_stats)
    # Save final stats summary to CSV
    stats_path = os.path.join(output_dir_for_corpus, "corpus_statistics.csv")
    stats_df.to_csv(stats_path, index=False)

