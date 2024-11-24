# How to Prepare Data for Data Analysis

Follow the steps below to prepare the data and run the analysis using Python:

---

## Step 1: Download and Install Required Tools
1. **Download the following files:**
   - **CzechFOI-DA** (project folder)
   - **DB Browser for SQLite** (from the official website)
   - **Vesely_106_202403141131.tar.xz** (CSV data)

2. **Install DB Browser for SQLite:**
   - Extract the contents of `DB Browser for SQLite` to `C:\github\CzechFOI-DA\DB.Browser.for.SQLite`.
   - Run `DB Browser for SQLite.exe` from this folder.

3. **Extract Vesely Data:**
   - Extract `Vesely_106_202403141131.tar.xz` to a CSV file: `Vesely_106_202403141131.csv`.

---

## Step 2: Create and Set Up the Database
1. **Create New Database:**
   - Open **DB Browser for SQLite**.
   - In the menu, go to **File > New Database** and name it `czechFO`.
   - Save the database file to: `C:\github\CzechFOI-DA\DB\czechFOI.db`.

2. **Open (Connect to) the New Database:**
   - Go to **File > Open Database** and select the newly created database: `C:\github\CzechFOI-DA\DB\czechFOI.db`.

3. **Import CSV into Table:**
   - In **DB Browser for SQLite**, go to **File > Import > Table from CSV File**.
   - Select the CSV file: `Vesely_106_202403141131.csv`.
   - Name the imported table `czech`.

4. **Check Imported Table:**
   - In the **Database Structure** tab, right-click on the table `czech` and select **Browse Table**.
   - Verify that the table has **11028372 rows**.

5. **Save the Database:**
   - Go to **File > Write Changes** and ignore any errors.
   - If prompted to save changes to the database file, click **Save**.

---

## Step 3: Run SQL Queries
1. **Open the SQL File:**
   - Open **DB Browser for SQLite**.
   - Go to **File > Open Database** and select the database: `C:\github\CzechFOI-DA\DB\czechFOI.db`.
   - In the menu, go to **Execute SQL**.
   - Click the **yellow Open SQL Query Symbol** and open the file: `C:\github\CzechFOI-DA\SQLQueries\All SQL Time.sql`.

2. **Execute SQL Queries:**
   - In the menu, click the **Execute all/selected SQL** button.
   - This will run the SQL queries (this may take about 30 minutes) to create all necessary tables and views for data analysis.

---

## Step 4: Export Pivot Views as CSV Files
1. **Check the Views:**
   - Go to the **Database Structure** tab.
   - Expand the **Views** entry. You should see **22 views**.

2. **Export Views as CSV Files:**
   - Right-click on each view and select **Export as CSV file**.
   - Use the standard export settings:
     - Column names in the first row
     - Field separator: `,`
     - Quote character: `"`, New line character: Windows `CR+LF`
   - Export all 22 views at once by selecting all in the **Export UI**.
   - Choose the **TERRA folder** as the location to save the CSV files.
   - Verify that you have **22 CSV files** in the TERRA folder.

3. **Close DB Browser for SQLite:**
   - If prompted, save the changes to the database file by clicking **Save**.

---

## Step 5: Set Up the Python Environment
1. **Install Python and Visual Studio Code:**
   - Download and install **Python** and **Visual Studio Code**.
   - Open **Visual Studio Code** after installation.

2. **Open the Python Script:**
   - Go to **File > Open** and select the Python script: `C:\github\CzechFOI-DA\Py Scripts\AH) 2D 6-Axis age-compare rolling-mean significance-1D-2D same-scale.py`.

---

## Step 6: Install Python Dependencies
To run the script successfully, you must install external Python packages (dependencies).

1. Open the terminal or command prompt in **Visual Studio Code**.
2. Run the following `pip` commands one by one to install the necessary libraries:

   ```bash
   pip install pandas
   pip install plotly
   pip install numpy
   pip install xarray
   pip install scipy
   pip install matplotlib
