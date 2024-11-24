### CzechFOI-DA

**Czech FOI Data Analysis** is a solution for analyzing, calculating, and visualizing FOI (Freedom of Information) data efficiently. It's faster and more flexible than using spreadsheets but requires some IT knowledge.

The [Python Scripts](https://github.com/gitfrid/CzechFOI-DA/tree/main/Py%20Scripts) process and visualize CSV data from the [TERRA folder](https://github.com/gitfrid/CzechFOI-DA/tree/main/TERRA), generating interactive HTML plots. Each plot compares two age groups. To interact with the plots, click on a legend entry to show/hide curves.

Download the processed plots for analysis from the [Plot Results Folder](https://github.com/gitfrid/CzechFOI-DA/tree/main/Plot%20Results).
Or simply adapt the Python scripts to meet your own analysis requirements!

Dates are counted as the number of days since January 1, 2020, for easier processing. "AGE_2023" represents age on January 1, 2023. The data can optionally be normalized per 100,000 for comparison.

Access the original Czech FOI data from a [Freedom of Information request](https://github.com/PalackyUniversity/uzis-data-analysis/blob/main/data/Vesely_106_202403141131.tar.xz).
To learn how the Pivot CSV files in the TERRA folder were created, see the [wiki](https://github.com/gitfrid/CzechFOI-DA/wiki)

### Software Requirements:
- [Python 3.12.5](https://www.python.org/downloads/) to run the scripts.
- [Visual Studio Code 1.92.2](https://code.visualstudio.com/download) to edit and run scripts.
- [Optional - DB Browser for SQLite 3.13.0](https://sqlitebrowser.org/dl/) for database creation, SQL queries, and CSV export.

### Disclaimer:
**The results have not been checked for errors. Neither methodological nor technical checks or data cleansing have been performed.**
