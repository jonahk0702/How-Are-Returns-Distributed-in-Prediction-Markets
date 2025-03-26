import os
from google.cloud import bigquery
import pandas as pd
import pickle

import pandas as pd
import numpy as np

import ast


import ast
import pickle


import db_dtypes



print("db-dtypes is installed correctly.")

#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "st312-442610-b9934d6c7287.json"
#client = bigquery.Client()



def make_csv_files():
    query = """
    SELECT `hash`
    FROM `bigquery-public-data.crypto_polygon.transactions`
    WHERE to_address = "0x78769d50be1763ed1ca0d5e878d93f05aabff29e"
    AND block_timestamp BETWEEN "2024-07-01 00:00:00" AND "2024-08-31 23:59:59"
    """

    transaction_hashes = client.query(query).to_dataframe()


    # Step 2 - get all the logs for each transactions


    #### Assumption 1 
    #* only use orders matched and that all orders matached are 0x63bf4d16b7fa898ef4c4b2b6d90fd201e9c56313b65638af6088d149d2ce956c

    import time
    batch_size = 10000
    matched_df = pd.DataFrame()
    filled_df = pd.DataFrame()




    batch_size = 10000

    for i in range(0, len(transaction_hashes), batch_size):
        # Get the current batch of 100 transaction hashes
        batch_hashes = transaction_hashes['hash'][i:i + batch_size].tolist()

        # Create the IN clause with the batch of hashes
        hashes_in_clause = "', '".join(batch_hashes)



        
        # Prepare the query with the IN clause
        query = f"""
        SELECT transaction_hash, address, `data`, topics, block_timestamp
        FROM `bigquery-public-data.crypto_polygon.logs`
        WHERE transaction_hash IN ('{hashes_in_clause}')
        AND block_timestamp BETWEEN "2024-07-01 00:00:00" AND "2024-08-31 23:59:59"

        """


        
        # Execute the query


        cur_logs = client.query(query).to_dataframe()


        
        # Filter the logs as needed
        orders_matched = cur_logs[cur_logs['topics'].apply(lambda x: x[0] == "0x63bf4d16b7fa898ef4c4b2b6d90fd201e9c56313b65638af6088d149d2ce956c")]
        orders_filled = cur_logs[cur_logs['topics'].apply(lambda x: x[0] == "0xd0a08e8c493f9c94f29311604c9de1b4e8c8d4c06bd0c789af57f2d65bfec0f6")]


        # Concatenate the results
        matched_df = pd.concat([matched_df, orders_matched], ignore_index=True)
        filled_df = pd.concat([filled_df, orders_filled], ignore_index=True)

        time.sleep(5)
        print(f"Processed {i + batch_size} transactions")



    matched_df.to_csv("./data/matched_orders.csv", index=False)
    filled_df.to_csv("./data/filled_orders.csv", index=False)
    print("Data has been saved to CSV files.")




def handle_orders(df):

    maker = df.iloc[0]['maker']
    taker = df.iloc[0]['taker']
    maker_asset_id = df.iloc[0]['maker_asset_id']
    taker_asset_id = df.iloc[0]['taker_asset_id']
    maker_amount = df.iloc[0]['maker_amount']
    taker_amount = df.iloc[0]['taker_amount']

    timeStamp = df.iloc[0]['block_timestamp']




    people.potential_add_person(maker)
    people.potential_add_person(taker)



    if maker_asset_id == "0000000000000000000000000000000000000000000000000000000000000000":
        ### Maker is buying!

        people.people[maker].buy(taker_asset_id, taker_amount, maker_amount, timeStamp)
        people.people[taker].sell(taker_asset_id, taker_amount, maker_amount, timeStamp)

        stockExchange.add_price(taker_asset_id, timeStamp, maker_amount/taker_amount)

    if taker_asset_id == "0000000000000000000000000000000000000000000000000000000000000000":
        # Taker is selling!

        people.people[taker].buy(maker_asset_id,maker_amount, taker_amount, timeStamp)
        people.people[maker].sell(maker_asset_id, maker_amount, taker_amount, timeStamp)
        stockExchange.add_price(maker_asset_id, timeStamp, taker_amount/maker_amount)





def process_poly():
    matched_df = pd.read_csv("./data/matched_orders.csv")
    filled_df = pd.read_csv("./data/filled_orders.csv")


    print("run")
    # Preprocess `filled_df` once
    filled_df = filled_df.copy()
    filled_df["topics_parsed"] = filled_df["topics"].apply(lambda x: ast.literal_eval(x))
    filled_df["takeOrderHash"] = filled_df["topics_parsed"].apply(lambda x: x[0][66:132])
    filled_df["maker"] = filled_df["topics_parsed"].apply(lambda x: x[0][132:198])
    filled_df["taker"] = filled_df["topics_parsed"].apply(lambda x: x[0][198:272])
    filled_df["maker_asset_id"] = filled_df["data"].str[2:66]
    filled_df["taker_asset_id"] = filled_df["data"].str[66:130]
    filled_df["maker_amount"] = filled_df["data"].str[130:194].apply(lambda x: int(x, 16))
    filled_df["taker_amount"] = filled_df["data"].str[194:258].apply(lambda x: int(x, 16))

    # Drop unnecessary columns to save memory
    filled_df = filled_df.drop(columns=["topics_parsed", "topics"])
    print("run")



    matched_df = matched_df.copy()
    matched_df["topics_parsed"] = matched_df["topics"].apply(lambda x: ast.literal_eval(x))
    matched_df["taker_order_hash"] = matched_df["topics_parsed"].apply(lambda x: x[0][66:132])
    matched_df["maker_asset_id"] = matched_df["data"].str[2:66]
    matched_df["taker_asset_id"] = matched_df["data"].str[66:130]
    matched_df["maker_amount"] = matched_df["data"].str[130:194].apply(lambda x: int(x, 16))
    matched_df["taker_amount"] = matched_df["data"].str[194:258].apply(lambda x: int(x, 16))

    # Drop unnecessary columns
    matched_df = matched_df.drop(columns=["topics_parsed", "topics"])

    print("run")



    # Merge on `taker_order_hash`
    merged_df = matched_df.merge(
        filled_df,
        left_on="taker_order_hash",
        right_on="takeOrderHash",
        how="inner"
    )

    # Drop redundant columns
    merged_df = merged_df.drop(columns=["takeOrderHash"])




    merged_df.to_csv("./data/merged_df.csv")

    merged_df = pd.read_csv("./data/merged_df.csv")



    for i in range(151):
        print(f"from {i*10000} to {(i+1)*10000}")

        for index, row in merged_df.iloc[i*10000:(i+1)*10000].iterrows():
            subset_df = pd.DataFrame([{
                "maker": row["maker"],  # From filled_df
                "taker": row["taker"],  # From filled_df
                "maker_asset_id": row["maker_asset_id_y"],
                "taker_asset_id": row["taker_asset_id_y"],
                "maker_amount": row["maker_amount_y"],
                "taker_amount": row["taker_amount_y"],
                "block_timestamp": row["block_timestamp_x"]
            }])

            handle_orders(subset_df)


        print("done!")


    with open("people_all.pkl", "wb") as f:
            pickle.dump(people, f)

    with open("stocks_all.pkl", "wb") as f:
        pickle.dump(stockExchange, f)