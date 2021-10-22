import time
import pandas as pd


def hamming_distance(i1, i2, numNodes):
    if len(i1) != len(i2):
        return 0

    z1, z2 = [0] * numNodes,  [0] * numNodes
    distance = 0

    # khoi tao z1, z2
    for i in range (0, len(i1)-1):
        if i1[i] <= numNodes and i1[i+1] <= numNodes:
            z1[i1[i]-1] = i1[i+1]
        elif i1[i] <= numNodes or i1[i+1] <= numNodes:
            index = min(i1[i], i1[i+1])
            z1[index-1] = -1
        if i == len(i1)-2 and i1[i+1] <= numNodes:
            z1[i1[i+1]-1] = -1

    for i in range (0, len(i2)-1):
        if i2[i] <= numNodes and i2[i+1] <= numNodes:
            z2[i2[i]-1] = i2[i+1]
        elif i2[i] <= numNodes or i2[i+1] <= numNodes:
            index = min(i2[i], i2[i+1])
            z2[index-1] = -1
        if i == len(i2)-2 and i2[i+1] <= numNodes:
            z2[i2[i+1]-1] = -1

    for i in range(numNodes):
        if z1[i] != z2[i]:
            distance += 1

    return distance

def to_excel(filename, data_ws, df, start_row=2, start_col=2):
    """Replacement for pandas .to_excel(). 
    For .xlsx and .xls formats only.
    args:
        start_row: df row +2; does not include header and is 1 based indexed.
    """
    writer = pd.ExcelWriter(filename.lower(), engine='openpyxl')
    import openpyxl
    try:
        wb = openpyxl.load_workbook(filename)
    except FileNotFoundError:
        wb = openpyxl.Workbook()
    if data_ws not in wb.sheetnames:
        wb.create_sheet(data_ws)

    # Create the worksheet if it does not yet exist.
    writer.book = wb
    writer.sheets = {x.title: x for x in wb.worksheets}

    ws = writer.sheets[data_ws]
    # Fill with blanks.
    try:
        for row in ws:
            for cell in row:
                cell.value = None
    except TypeError:
        pass

    # Write manually to avoid overwriting formats.

    # Column names.
    ws.cell(1, 1).value = df.columns.name
    for icol, col_name in zip(range(2, len(df.columns) + 2), df.columns):
        ws.cell(1, icol).value = col_name

    # Row headers.
    for irow, row_name in zip(range(2, len(df.index) + 2), df.index):
        ws.cell(irow, 1).value = row_name

    # Body cells.
    for row, irow in zip([x[1] for x in df.iloc[start_row - 2:].iterrows()], list(range(start_row, len(df.index) + 2))):
        for cell, icol in zip([x[1] for x in row.iloc[start_col - 2:].items()], list(range(start_col, len(df.columns) + 2))):
            ws.cell(irow, icol).value = cell  # Skip the index.

    for row in ws.values:
        print('\t'.join(str(x or '') for x in row))
    print('Saving.')
    while True:
        try:
            writer.save()
            break
        except PermissionError:
            print(f'Please close {filename} before we can write to it!')
            time.sleep(2)
    writer.close()
    print('Done saving df.')


def save_solution(instance, run_time, number_of_tech, cost, work_time, version):
    # Instance, run_time, number_of_tech, cost, work_time
    data=[]
    try:
        df = pd.read_excel('result.xlsx', sheet_name='result')
        
        for i in range(len(df['Instance'])):
            data.append([
                df['Instance'][i], df['run_time'][i], df['number_of_tech'][i], df['cost'][i], df['work_time'][i], df['version'][i]
            ])
    except FileNotFoundError:
        print("No such file or directory: 'result.xlsx'")
    data.append([instance, run_time, number_of_tech, cost, work_time, version])
    df = pd.DataFrame(data, columns=['Instance', 'run_time', 'number_of_tech', 'cost', 'work_time', 'version'])
    to_excel('result.xlsx', 'result', df)


if __name__ == "__main__":
    i1 = [2, 8, 5, 7, 3, 4, 6, 1]
    i2 = [6, 3, 4, 2, 8, 7, 1, 5]
    ds = hamming_distance(i1, i2, 6)
    print(ds)