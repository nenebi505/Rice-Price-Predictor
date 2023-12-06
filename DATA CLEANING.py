import pandas as pd

def load_and_clean_table(file_path, sheet_name):
    # Load the data, starting from row 3 to include the column headers
    data = pd.read_excel(file_path, sheet_name, header=2)

    # Melt the dataframe to convert it into a long format
    # The month information is in the first column (index 0)
    data_melted = data.melt(id_vars=[data.columns[0]], var_name='Year', value_name='Price')

    # Remove any rows with NaN in 'Price'
    data_melted = data_melted.dropna(subset=['Price'])

    # Create a 'Date' column by combining 'Year' and the month column

def load_and_clean_table(file_path, sheet_name):
    # Load the data, starting from row 3 to include the column headers
    data = pd.read_excel(file_path, sheet_name, header=2)

    # Melt the dataframe to convert it into a long format
    month_column = data.columns[0]  # First column for month
    data_melted = data.melt(id_vars=[month_column], var_name='Year', value_name='Price')

    # Remove any rows with NaN in 'Price'
    data_melted = data_melted.dropna(subset=['Price'])

    # Create a 'Date' column by properly combining the month and year
    # The year should be in the format YYYY, so let's extract it from the 'Year' column
    data_melted['Year'] = data_melted['Year'].str.split('/').str[0]

def load_and_clean_table(file_path, sheet_name):
    # Load the data, starting from row 3 to include the column headers
    data = pd.read_excel(file_path, sheet_name, header=2)

    # Melt the dataframe to convert it into a long format
    month_column = data.columns[0]  # First column for month
    data_melted = data.melt(id_vars=[month_column], var_name='Year', value_name='Price')

    # Debugging: Print the first few rows of data_melted
    print(data_melted.head())

    # Remove any rows with NaN in 'Price'
    data_melted = data_melted.dropna(subset=['Price'])

    # Debugging: Print the first few rows of data_melted after dropping NaNs
    print(data_melted.head())

    # Create a 'Date' column by properly combining the month and year
    data_melted['Year'] = data_melted['Year'].str.split('/').str[0]  # Extracting the first part of the 'Year'

def load_and_clean_table(file_path, sheet_name):
    # Load the data, starting from row 3 to include the column headers
    data = pd.read_excel(file_path, sheet_name, header=2)

    # Identify the correct 'Year' columns (skip 'Unnamed' and non-year columns)
    year_columns = [col for col in data.columns[1:] if "Unnamed:" not in col and '/cwt' not in col]

    # Filter out rows that do not represent months
    months = ['January', 'February', 'March', 'April', 'May', 'June', 
              'July', 'August', 'September', 'October', 'November', 'December']
    data = data[data[data.columns[0]].isin(months)]

    # Melt the dataframe to convert it into a long format
    data_melted = pd.melt(data, id_vars=[data.columns[0]], value_vars=year_columns, var_name='Year', value_name='Price')

    # Remove any rows with NaN in 'Price'
    data_melted = data_melted.dropna(subset=['Price'])

    # Extract the first part of the 'Year' and create a 'Date' column
    data_melted['Year'] = data_melted['Year'].str.split('/').str[0]
    data_melted['Date'] = pd.to_datetime(data_melted[data.columns[0]] + ' ' + data_melted['Year'])

    # Return the processed data
    return data_melted[['Date', 'Price']].sort_values(by='Date')

def main():
    file_path = r"C:\Users\DELL\Downloads\U_S_Rough_and_Milled_Rice_Prices_monthly_and_marketing_year (1).xls"

    # Process Table 15
    table_15_cleaned = load_and_clean_table(file_path, 'Table 15')
    table_15_cleaned.to_csv('Table_15_Cleaned.csv', index=False)

    # Process Table 16
    table_16_cleaned = load_and_clean_table(file_path, 'Table 16')
    table_16_cleaned.to_csv('Table_16_Cleaned.csv', index=False)

    print("Data processing complete. Files saved as 'Table_15_Cleaned.csv' and 'Table_16_Cleaned.csv'.")

if __name__ == "__main__":
    main()

def process_table(data):
    # Filter out rows that do not represent months
    months = ['January', 'February', 'March', 'April', 'May', 'June', 
              'July', 'August', 'September', 'October', 'November', 'December']
    data_filtered = data[data['Unnamed: 0'].isin(months)]

    # Melt the dataframe to convert it into a long format
    data_melted = data_filtered.melt(id_vars=['Unnamed: 0'], var_name='Year', value_name='Price')

    # Remove any rows with NaN in 'Price'
    data_melted = data_melted.dropna(subset=['Price'])

    # Extract the year part and handle the year format
    data_melted['Year'] = data_melted['Year'].str.extract(r'(\d{4})').astype(str)

    # Creating a 'Date' column by combining the month and year
    data_melted['Date'] = pd.to_datetime(data_melted['Unnamed: 0'] + ' ' + data_melted['Year'], errors='coerce')

    # Drop rows where Date couldn't be parsed
    data_melted.dropna(subset=['Date'], inplace=True)

    # Selecting relevant columns and sorting by date
    return data_melted[['Date', 'Price']].sort_values(by='Date')

    # Apply the processing to Table 15 and Table 16
    table_15_processed = process_table(table_15)
    table_16_processed = process_table(table_16)

    # Save the processed data to new CSV files and display the first few rows
    processed_file_path_15 = '/mnt/data/Table_15_Processed.csv'
    processed_file_path_16 = '/mnt/data/Table_16_Processed.csv'
    table_15_processed.to_csv(processed_file_path_15, index=False)
    table_16_processed.to_csv(processed_file_path_16, index=False)
    print(table_15_processed.head())
    print(table_16_processed.head())
    