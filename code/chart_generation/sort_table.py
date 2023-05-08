def sort_main_table():
    with open("data/table_data.txt") as fp:
        lines = fp.readlines()
        data = []
        for line in lines:
            line = line.strip()
            if line != "":
                values = line.split("&")
                values[4], values[5] = values[5], values[4]
                data.append(values)
        data = sorted(data, key=lambda x: int(x[2].strip().replace(',', '')), reverse=True)
        for line in data:
            print("&".join(line))


def sort_dataset_table():
    with open("data/dataset_data.txt") as fp:
        lines = fp.readlines()
        data = []
        for line in lines:
            line = line.strip()
            if line != "":
                values = line.split("&")
                data.append(values)
        data = sorted(data, key=lambda x: (int(x[2].strip().replace(',', '')), x[1].strip().lower()), reverse=True)
        for line in data:
            print("&".join(line))


if __name__ == "__main__":
    sort_dataset_table()