# This script is to deal with and arrange data

import numpy as np
import os.path as path
from pathlib import Path
import os
import pandas as pd
import sqlalchemy
from sqlalchemy.orm import declarative_base, relationship
import tkinter as tk
from tkinter import filedialog
import xlwings as xw
from xlwings.constants import DeleteShiftDirection

from pyprocesscontrol import statistics


class raw_data:
    """The raw data class contains the raw data for analysis

    This class will upload and group data in an excel document. The document can be uploaded to the class and the class will extract the data into a pandas dataframe. From there, the data can be grouped on levels to as tight or wide of a grouping as needed.

    Parameters
    ----------
    filename : str
        The location name of the excel document in the working directory

    Attributes
    ----------
    filename : str
    sheetname : str, default = 'Sheet1'
    header : int, default = 1
    index : int, default = 1
    groupby : array_like
    path : str

    """

    def __init__(self, filename: str, **kwargs):
        """ """
        self.filename = filename
        self.sheetname: str | None = kwargs.get("sheetname", "Sheet1")
        header: int = kwargs.get("header", 1)
        index: int = kwargs.get("index", 1)
        self.groupby: list = kwargs.get("groupby", None)
        self.path = None

        # look for the filename in the working directory
        for root, dir, files in os.walk(os.getcwd()):
            if self.filename in files:
                self.path = os.path.join(root, self.filename)

        # Not in the working directory -- You tell me where it is
        if self.path == None:
            try:
                root = tk.Tk()
                root.withdraw()

                file_path = filedialog.askopenfile()
                self.path = Path(file_path.name)
            except:
                raise ValueError("File could not be found")

        split_tup = os.path.splitext(filename)
        name = split_tup[0]
        extension = split_tup[1]

        # What is the format of the file
        match extension:
            case ".xlsx":
                wb = xw.Book(self.path)
                wb.activate()
                sheet = wb.sheets(self.sheetname)
                sheet.range("A1").select()
                self.df = xw.load(header=header, index=index)

            case ".xls":
                wb = xw.book(self.path)
                sheet = wb.sheets(self.sheetname)
                sheet.range("A1").select()
                self.df = xw.load(header=header, index=index)

            case ".csv":
                self.df = pd.read_csv(self.path)

        # Recursive grouping over the list given
        if self.groupby != None:
            self.group(groupby=self.groupby)

    def group(self, groupby: list) -> dict:
        """This function groups the raw data and exports the groups in a dictionary format.

        Parameters
        ----------
        groupby : list_like
            list of elements to group the dataframe by


        Returns
        -------
        dict
            dictionary of grouped or recursively grouped dataframes on the `groupby` parameter.
        """
        groups = {}

        if len(groupby) > 1:
            groups = self.__recursive_grouping(self.df, len(groupby), groupby)
        if len(groupby) == 1:
            for name, group in self.df.groupby(by=groupby[0], axis=1):
                groups[name] = group

        self.groups = groups
        self.keys = list(self.groups.keys())

    def __recursive_grouping(self, df: pd.DataFrame, number: int, level: list):
        r_dict = dict()

        if number > 1:
            for name, group in df.groupby(by=level[-number]):
                number = number - 1
                r_dict[name] = self.__recursive_grouping(
                    group, number=number, level=level
                )
        else:
            for name, group in df.groupby(by=level):
                j = 0
                group = group.dropna(axis=1, how="all")
                for column in group.columns:
                    num = 1
                    all_columns = list(group.columns)
                    if j == 0:
                        lst = [column]
                        j = 1
                        continue

                    temp_col = column
                    while column in lst:
                        column = temp_col + "_" + str(num)

                        if column in lst:
                            num += 1

                    lst.append(column)

                group.columns = lst
                r_dict[name] = group

            return r_dict

        return r_dict


class metadata:
    """A class containing metadata on the raw data class


    This class will carry the metadata for the raw_data. This will include spec limits, normality coefficients (shapiro-wilks test), and other metadata that will be useful in the process statisics.

    Parameters
    ----------
    filename : str
        The `filename` for the file containing raw data

    Attributes
    ----------
    filename
    header : int, default = 1
    index : int, default = 1
    groupby : array_like
    sheetname : str, default = 'Sheet1'
    raw_data : datastructures.raw_data
    metadata : pandas.DataFrame
    """

    def __init__(self, filename, **kwargs):
        self.header: int = kwargs.get("header", 1)
        self.index: int = kwargs.get("index", 1)
        self.groupby: list = kwargs.get("groupby", None)
        self.sheetname: str | None = kwargs.get("sheetname", None)

        self.raw_data = raw_data(
            filename,
            sheetname=self.sheetname,
            header=self.header,
            index=self.index,
            groupby=self.groupby,
        )

        self.load_data()

        columns = ["Product", "Grade", "Spec", "USL", "LSL", "TGT", "UCL", "LCL"]

        self.metadata = pd.DataFrame(columns=columns)

    def set_spec_limit(self, spec: str, usl: float, lsl: float, **kwargs):
        """
        This function sets the spec limits on a specific spec. Can also add grade code and product if needed

        Parameters
        ----------
            spec : str
                Column string for the spec to apply to
            usl : float
                upper spec limit for the spec
            lsl : float
                lower spec limit for the spec
        """

        product: str = kwargs.get("product", None)
        grade: str = kwargs.get("grade", None)

        try:
            if (
                self.metadata["Product"].loc[
                    (self.metadata["Product"] == product)
                    & (self.metadata["Grade"] == grade)
                ][0]
                != None
            ):
                self.metadata["USL"].loc[
                    (self.metadata["Product"] == product)
                    & (self.metadata["Grade"] == grade)
                ] = usl

                self.metadata["LSL"].loc[
                    (self.metadata["Product"] == product)
                    & (self.metadata["Grade"] == grade)
                ] = lsl
        except KeyError:
            try:
                data = self.raw_data.groups[product][product, grade][spec]
            except Exception:
                raise

            pd_dict = {
                "Product": product,
                "Grade": grade,
                "Spec": spec,
                "USL": usl,
                "LSL": lsl,
            }

            self.metadata.loc[len(self.metadata.index)] = pd_dict

    def create_new_metadata(self, func: str):
        """This function adds new metadata from a specified list to the table for tracking in specs

        Parameters
        ----------
        func : str
            A statistical function to apply to the metadata dataframe.
            The list of possible functions is:
                1.) 'shapiro-wilk' \n
                2.) 'cpk' \n
                3.) 'ppk' \n
                4.) 'cp' \n
                5.) 'pp' \n
                6.) 'defect-rate' \n

        Notes
        ----------
        1.) The shapiro-wilk test tests the dataset against the null hypothesis that the data comes from a normally distributed dataset. By default the alpha is 0.05. If the shapiro-wilk value is less than 0.05 one cannot confirm the null hypothesis and the samples may not come from a normally distributed data set.

        2.) Both cpk and cp require a normally distributed, well controlled data set to be accurate. Do not use cpk and cp as a signal of control until the process is under control and normally distributed

        3.) Both ppk and pp reject the notion the process is under control and the results are normally distributed. These process capability statistics should be used when the process is not under control

        4.) The 'defect-rate' function requires a normally distributed process to accurately predict the failure rate.

        """

        match func:
            case "shapiro-wilk":
                for index, data in self.metadata.iterrows():
                    self.__metadata_apply(
                        func,
                        product=data["Product"],
                        grade=data["Grade"],
                        spec=data["Spec"],
                    )
            case "cpk":
                for index, data in self.metadata.iterrows():
                    self.__metadata_apply(
                        func,
                        product=data["Product"],
                        grade=data["Grade"],
                        spec=data["Spec"],
                    )
            case "ppk":
                for index, data in self.metadata.iterrows():
                    self.__metadata_apply(
                        func,
                        product=data["Product"],
                        grade=data["Grade"],
                        spec=data["Spec"],
                    )
            case "cp":
                for index, data in self.metadata.iterrows():
                    if index == 53:
                        pass

                    self.__metadata_apply(
                        func,
                        product=data["Product"],
                        grade=data["Grade"],
                        spec=data["Spec"],
                    )
            case "pp":
                for index, data in self.metadata.iterrows():
                    self.__metadata_apply(
                        func,
                        product=data["Product"],
                        grade=data["Grade"],
                        spec=data["Spec"],
                    )
            case "defect-rate":
                for index, data in self.metadata.iterrows():
                    self.__metadata_apply(
                        func,
                        product=data["Product"],
                        grade=data["Grade"],
                        spec=data["Spec"],
                    )

    def save_data(
        self,
        export_type: str = "xlsx",
        save_metadata: bool = True,
        save_rawdata: bool = True,
        create_file: bool = False,
        append_data: bool = True,
    ):
        """
        This function exports the broken down raw data and meta data into documents and saves them.

        Args:
            export_type: how the data should be exported. Either xlsx or csv (future version can save database)
        """

        cwd = os.getcwd()

        if create_file == True:
            path = os.path.join(os.getcwd(), "data_files")
            try:
                os.mkdir(path)
            except FileExistsError:
                pass

            os.chdir(path)

        match export_type:
            case "xlsx":
                try:
                    self.__save_raw_data(
                        save_rawdata=save_rawdata,
                        append_data=append_data,
                    )
                except:
                    pass

                if save_metadata == True:
                    try:
                        self.__save_metadata(
                            append_data=append_data,
                            save_metadata=save_metadata,
                        )
                    except:
                        pass

            case "sql":
                # insert code to use sqlalchemy to store all of the data
                pass

        os.chdir(cwd)

    def load_data(self):
        """
        This function manages the loading of the _raw_data and _metadata files if they exist.

        Args:
            None

        Returns:
            None

        """
        return_dict = {}
        temp_dict = {}

        wb = None
        meta_wb = None

        cwd = os.getcwd()

        raw_data = "_raw_data.xlsx"
        meta = "_metadata.xlsx"
        groupby = self.groupby

        if len(self.groupby) == 1:
            groupby = self.groupby[0]

        try:
            os.chdir(os.path.join(os.getcwd(), "data_files"))
        except:
            pass

        for root, dirs, files in os.walk(os.getcwd()):
            if raw_data in files:
                path = os.path.join(root, raw_data)
                wb = xw.Book(path)
                break

        for root, dirs, files in os.walk(os.getcwd()):
            if meta in files:
                meta_path = os.path.join(root, meta)
                meta_wb = xw.Book(meta_path)
                break

        os.chdir(cwd)

        if wb == None and meta_wb == None:
            return

        self.__load_raw_data(wb)
        self.__load_metadata(meta_wb=meta_wb)

    def __load_raw_data(self, wb: xw.Book):
        groupby = self.groupby
        temp_dict = {}
        return_dict = {}

        if wb == None:
            return

        wb.activate()

        self.raw_data.keys = list(set(self.raw_data.keys + wb.sheet_names))

        for key in self.raw_data.keys:
            groups = None
            try:
                sheet = wb.sheets[key]
                sheet.activate()
                data: pd.DataFrame = xw.load()
                groups = data.groupby(groupby)
            except:
                pass

            temp_dict = {}
            for grouping in self.raw_data.groups[key].keys():
                df_raw_data = self.raw_data.groups[key][grouping].copy()

                if groups is not None:
                    try:
                        temp_df = groups.get_group(grouping)
                        indexes = temp_df[~temp_df.index.isin(df_raw_data.index)]
                        typ = temp_df.index.dtype
                        df_raw_data.index = df_raw_data.index.astype(typ)
                        temp_dict[grouping] = pd.concat(
                            [
                                df_raw_data,
                                temp_df.loc[indexes.index].reset_index(inplace=True),
                            ]
                        ).copy()
                    except KeyError:
                        temp_dict[grouping] = df_raw_data
                else:
                    temp_dict[grouping] = df_raw_data

            return_dict[key] = temp_dict

        self.raw_data.groups = return_dict

    def __load_metadata(self, meta_wb: xw.Book):
        if meta_wb == None:
            return

        meta_wb.activate()
        sheets = meta_wb.sheets[meta_wb.sheet_names[0]]
        sheets.activate()
        sheets.range("A1").select()
        meta_data = xw.load()

        self.metadata = meta_data

    def __metadata_apply(self, func, **kwargs):
        """
        The purpose of this function is to apply calculations to rows of data and pass them into the metadata dataframe. This will make custom calculations possible for metadata

        Args:
            func: A function to apply to a dataframe column and the result will be returned to the metadata column
            col_name: The name of the column the formula will apply on
        """

        grade: str = kwargs.get("grade", None)
        product: str = kwargs.get("product", None)
        spec: str = kwargs.get("spec", None)

        if self.raw_data.groups != None:
            try:
                data = self.raw_data.groups[product][product, grade][spec]
                data = pd.to_numeric(data, errors="coerce")
            except KeyError:
                data = "No Data"

        if isinstance(data, str) == True:
            try:
                self.metadata.loc[
                    (self.metadata["Grade"] == grade)
                    & (self.metadata["Product"] == product)
                    & (self.metadata["Spec"] == spec),
                    func,
                ] = data
                return
            except KeyError:
                self.metadata["shapiro-wilk"] = np.empty(len(self.metadata.index))
                self.metadata.loc[
                    (self.metadata["Grade"] == grade)
                    & (self.metadata["Product"] == product)
                    & (self.metadata["Spec"] == spec),
                    func,
                ] = data
                return

        match func:
            case "shapiro-wilk":
                if isinstance(data, pd.Series):
                    x = statistics.shapiro_wilk(data)

            case "cpk":
                if isinstance(data, pd.Series):
                    x = statistics.cpk(
                        data,
                        usl=self.metadata["USL"].loc[
                            (self.metadata["Grade"] == grade)
                            & (self.metadata["Product"] == product)
                            & (self.metadata["Spec"] == spec)
                        ],
                        lsl=self.metadata["LSL"].loc[
                            (self.metadata["Grade"] == grade)
                            & (self.metadata["Product"] == product)
                            & (self.metadata["Spec"] == spec)
                        ],
                    )

            case "cp":
                if isinstance(data, pd.Series):
                    x = statistics.cp(
                        data,
                        usl=self.metadata["USL"].loc[
                            (self.metadata["Grade"] == grade)
                            & (self.metadata["Product"] == product)
                            & (self.metadata["Spec"] == spec)
                        ],
                        lsl=self.metadata["LSL"].loc[
                            (self.metadata["Grade"] == grade)
                            & (self.metadata["Product"] == product)
                            & (self.metadata["Spec"] == spec)
                        ],
                    )

            case "pp":
                if isinstance(data, pd.Series):
                    x = statistics.pp(
                        data,
                        usl=self.metadata["USL"].loc[
                            (self.metadata["Grade"] == grade)
                            & (self.metadata["Product"] == product)
                            & (self.metadata["Spec"] == spec)
                        ],
                        lsl=self.metadata["LSL"].loc[
                            (self.metadata["Grade"] == grade)
                            & (self.metadata["Product"] == product)
                            & (self.metadata["Spec"] == spec)
                        ],
                    )

            case "ppk":
                if isinstance(data, pd.Series):
                    x = statistics.ppk(
                        data,
                        usl=self.metadata["USL"].loc[
                            (self.metadata["Grade"] == grade)
                            & (self.metadata["Product"] == product)
                            & (self.metadata["Spec"] == spec)
                        ],
                        lsl=self.metadata["LSL"].loc[
                            (self.metadata["Grade"] == grade)
                            & (self.metadata["Product"] == product)
                            & (self.metadata["Spec"] == spec)
                        ],
                    )

            case "defect-rate":
                if isinstance(data, pd.Series):
                    x = statistics.defect_probability(
                        data,
                        usl=self.metadata["USL"].loc[
                            (self.metadata["Grade"] == grade)
                            & (self.metadata["Product"] == product)
                            & (self.metadata["Spec"] == spec)
                        ],
                        lsl=self.metadata["LSL"].loc[
                            (self.metadata["Grade"] == grade)
                            & (self.metadata["Product"] == product)
                            & (self.metadata["Spec"] == spec)
                        ],
                    )

        # does column func already exist
        try:
            self.metadata.loc[
                (self.metadata["Grade"] == grade)
                & (self.metadata["Product"] == product)
                & (self.metadata["Spec"] == spec),
                func,
            ] = x

        # Add func column if it does not exist
        except KeyError:
            self.metadata[func] = np.empty(len(self.metadata.index))
            self.metadata.loc[
                (self.metadata["Grade"] == grade)
                & (self.metadata["Product"] == product)
                & (self.metadata["Spec"] == spec),
                func,
            ] = x

    def __save_raw_data(
        self,
        save_rawdata: bool = True,
        append_data: bool = True,
    ):
        """This function saves the rawdata to be held in the metadata class"""

        if save_rawdata == True:
            path = os.path.join(os.getcwd(), "_raw_data.xlsx")
            try:
                wb = xw.Book(path)
            except:
                wb = xw.Book()

            # os.chdir(cwd)

            # Create new sheet for each product
            for key in self.raw_data.keys:
                try:
                    wb.sheets.add(key)
                    j = 0
                    sheet = wb.sheets[key]
                except:
                    sheet = wb.sheets[key]

                sheet.activate()

                if append_data == True:
                    sheet.range("A1").select()
                    try:
                        data = xw.load()
                        data = data.dropna(axis=1, how="all")
                        sheet.clear()
                        groupings = data.groupby("Grade Code")
                        j = 0
                    except:
                        j = 0
                        groupings = self.raw_data.df[
                            self.raw_data.df[self.groupby[0]] == key
                        ].groupby(self.groupby[1])
                        sheet.range("A1").select()

                # group all grades together but don't paste over them
                add_data = pd.DataFrame()
                for grade in groupings.groups.keys():
                    # Is this the first pass through?
                    if j == 0:
                        try:
                            data = self.raw_data.groups[key][(key, grade)]
                            data.index = pd.to_datetime(data.index)
                            temp_df = groupings.get_group(grade)
                            temp_df = temp_df[~temp_df.index.isin(data.index)]
                            add_data = pd.concat(
                                [
                                    add_data,
                                    data,
                                    temp_df.reset_index(inplace=True),
                                ]
                            )
                        except:
                            add_data = pd.concat([add_data, data])
                        sheet.range("A1").select()
                        # sheet["A1"].value = add_data
                        j = 1
                    else:
                        try:
                            data = self.raw_data.groups[key][(key, grade)]
                            temp_df = groupings.get_group(grade)
                            temp_df = temp_df[~temp_df.index.isin(data.index)]
                            add_data = pd.concat(
                                [
                                    add_data,
                                    data,
                                    temp_df.reset_index(inplace=True),
                                ]
                            )
                        except:
                            add_data = pd.concat([add_data, data])

                sheet.range("A1").options(header=True).value = add_data

            # save in working directory
            path = os.path.join(os.getcwd(), "_raw_data.xlsx")

            wb.save(path=path)

    def __save_metadata(
        self,
        save_metadata: bool = True,
        append_data: bool = True,
    ):
        """This function saves the metadata"""

        path = os.path.join(os.getcwd(), "_metadata.xlsx")
        try:
            meta_wb = xw.Book(path)
        except:
            meta_wb = xw.Book()

        meta_wb.activate()
        sheet = meta_wb.sheets[meta_wb.sheet_names[0]]
        sheet.range("A1").select()

        try:
            curr_metadata = xw.load(index=1, header=1)
            curr_metadata = curr_metadata[
                ~curr_metadata.index.isin(self.metadata.index)
            ]
            sheet.range("A1").value = pd.concat([self.metadata, curr_metadata])
        except:
            sheet.range("A1").value = self.metadata

        # save in working directory
        meta_wb.save(path)


class data_structure:
    """
    The purpose of this class is to break down the imported data into both
    product and grade code. Then, store the data in an easily accessable way
    to return specific data quickly. This may evolve to include a database as well if I feel like it needs it.
    """

    def __init__(self, workbook: str):
        if path.isfile(workbook) == True:
            self.wb = xw.Book(workbook)
            data = self.__data_extract(self.wb)
            levels = ["Product Name", "Grade Code"]
            self.groups = self.recursive_grouping(data, len(levels), levels)
        else:
            raise ValueError("The workbook does not exist")

    def __data_extract(self, wb: xw.Book) -> pd.DataFrame:
        """
        This function extracts data out of a specific workbook and places
        it into a pandas dataframe. This will only work with the quality
        dashboard excel spreadsheet.

        Args:
            wb: xlwings workbook associated with the data to be extracted

        Returns:
            pandas DataFrame
        """

        sheet = wb.sheets[wb.sheet_names[0]]
        sheet.range((4, 1), (267, 176)).select()
        sheet.range("4:287").api.Delete(DeleteShiftDirection.xlShiftUp)
        val = sheet.range("A4:C4").value
        sheet.range("A4:C4").clear()
        sheet.range("A2:C2").value = val
        sheet.range("3:4").api.Delete(DeleteShiftDirection.xlShiftUp)
        sheet.range("1:1").api.Delete(DeleteShiftDirection.xlShiftUp)
        sheet.range("D:D").api.Delete(DeleteShiftDirection.xlShiftToLeft)
        sheet.range("A1").select()
        data = xw.load(header=1, index=1)
        wb.close()
        return data

    def data_grouping(self, data, level, number) -> dict:
        """
        This function groups the data into levels and returns each level in a dictionary.
        """

        return_dict = dict()

        length = len(level)

        for i in range(0, len(level)):
            pass

    def recursive_grouping(self, y, number, level):
        r_dict = dict()

        if number > 1:
            for name, group in y.groupby(by=level[-number], axis=0):
                number = number - 1
                r_dict[name] = self.recursive_grouping(
                    group, number=number, level=level
                )
        else:
            for name, group in y.groupby(by=level, axis=0):
                r_dict[name] = group

            return r_dict

        return r_dict


class sql_query:
    """
    This class is designed to store the control chart data into the database. This will also store all the KPI data
    """

    def __init__(self):
        cwd = os.getcwd()
        path = path.join(cwd, "chart_data")
        connected = False

        while connected != True:
            self.engine = sqlalchemy.create_engine("sqlite:///" + path)

            try:
                self.engine.connect()
                connected = True

            except:
                print("an error has occurred")

        self.Base = declarative_base()

        Session = sqlalchemy.orm.sessionmaker(bine=self.engine)
        self.Session = Session()

    def import_data(self, data: pd.DataFrame) -> None:
        """
        This function imports and stores new data into the database

        Args:
            data: pandas dataframe containing the data to be stored
        """

        pass
