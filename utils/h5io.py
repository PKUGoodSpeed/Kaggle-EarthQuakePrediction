import pandas as pd
import tables

def write_to_h5(filename, df):
    print('Writing dataframe into {0}...'.format(filename))
    h5file = tables.open_file(filename, 'w', driver='H5FD_CORE')
    record = df.astype(float).to_records(index=False)
    filters= tables.Filters(complevel=5)        ## setting compression level
    h5file.create_table(h5file.root, 'DATA', filters=filters, obj=record)
    h5file.close()
    
def load_from_h5(filename, table='/DATA'):
    print("Loading dataframe from {0}...".format(filename))
    with tables.open_file(filename) as fp:
        df = pd.DataFrame(fp.get_node(table).read())
    return df
