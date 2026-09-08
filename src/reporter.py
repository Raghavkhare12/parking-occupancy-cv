from pathlib import Path

import pandas as pd


class ParkingReportGenerator:
    """
    Generates reports for parking-lot occupancy analysis.
    """

    def __init__(self, output_directory):
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(
            parents=True,
            exist_ok=True
        )

    def save_csv(self, results_df, filename="parking_analysis.csv"):
        """
        Save individual parking-space predictions to CSV.
        """

        output_path = self.output_directory / filename

        results_df.to_csv(
            output_path,
            index=False
        )

        return output_path

    def create_summary_dataframe(self, summary):
        """
        Convert summary statistics into a DataFrame.
        """

        return pd.DataFrame([
            {
                "total_spaces": summary["total_spaces"],
                "occupied_spaces": summary["occupied_spaces"],
                "empty_spaces": summary["empty_spaces"],
                "occupancy_percentage": round(
                    summary["occupancy_percentage"],
                    2
                )
            }
        ])

    def save_summary(
        self,
        summary,
        filename="parking_summary.csv"
    ):
        """
        Save overall occupancy statistics.
        """

        output_path = self.output_directory / filename

        summary_df = self.create_summary_dataframe(
            summary
        )

        summary_df.to_csv(
            output_path,
            index=False
        )

        return output_path