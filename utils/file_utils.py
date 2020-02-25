from typing import List
import json
import gzip
import csv


def write_items(items: List[str], output_file):
    with open(output_file, 'w') as f:
        for item in items:
            f.write(str(item) + "\n")
    f.close()


def read_lines(input_file: str) -> List[str]:
    lines = []
    with open(input_file, "rb") as f:
        for l in f:
            lines.append(l.decode().strip())
    return lines


def read_jsonl_lines(input_file: str) -> List[dict]:
    with open(input_file) as f:
        lines = f.readlines()
        return [json.loads(l.strip()) for l in lines]

class TsvIO(object):
    @staticmethod
    def read(filename, known_schema=None, sep="\t", gzipped=False, source=None):
        """
        Read a TSV file with schema in the first line.
        :param filename: TSV formatted file
        :param first_line_schema: True if the first line is known to contain the schema of the
        tsv file. False by default.
        :param sep: Separator used in the file. Default is '\t`
        :return: A list of data records where each record is a dict. The keys of the dict
        correspond to the column name defined in the schema.
        """
        first = True

        if gzipped:
            fn = gzip.open
        else:
            fn = open

        line_num = 0

        with fn(filename, 'rt') as f:
            for line in f:
                if first and known_schema is None:
                    first = False
                    known_schema = line.split(sep)
                    known_schema = [s.strip() for s in known_schema]
                else:
                    line_num += 1
                    data_fields = line.split(sep)
                    data = {k.strip(): v.strip() for k, v in zip(known_schema, data_fields)}
                    data['source'] = filename if source is None else source
                    data['line_num'] = line_num
                    yield data
        f.close()

    @staticmethod
    def make_str(item, sub_sep="\t"):
        if isinstance(item, list):
            return sub_sep.join([TsvIO.make_str(i) for i in item])
        else:
            return str(item)

    @staticmethod
    def write(records: List[dict], filename, schema=None, sep='\t', append=False, sub_sep=';'):
        """
        Write a TSV formatted file with the provided schema
        :param records: List of records to be written to the file
        populated
        :param filename: Output filename
        :param schema: Order in which fields from the Sentence object will be written
        :param sep: Separator used in the file. Default is '\t`
        :param append: Whether to use append mode or write a new file
        :param sub_sep: If a field contains a list of items in JSON, this seperator will be used
        to separate values in the list
        :return:
        """
        mode = 'a' if append else 'w'

        if sep == "\t":
            with open(filename, mode) as f:
                if schema is not None and not append:
                    f.write(sep.join(schema) + "\n")
                for record in records:
                    f.write(sep.join([TsvIO.make_str(record.__getitem__(field), sub_sep=sub_sep) for
                                      field in schema]))
                    f.write('\n')
            f.close()
        elif sep == ",":
            with open(filename, mode) as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=schema)
                writer.writeheader()
                for record in records:
                    writer.writerow(record)
            csvfile.close()