import pathlib

import pandas as pd
import dropbox

access_token = "sl.B5Hwk7Wsy_yaYUtPExrWkHUTPb06dr8dao32xsboUFSA2DKi3wJ6Wkmkxt-Y994xqfPfJKBxFQgsijynHCRuSvyd-sVWCEajpdVBbaxjf4ePCxQXh9pHdcpvhWnonsmHT5EJODDcdTIV"


def upload_to_dropbox(local_file_path, dropbox_file_path, access_token):
    dbx = dropbox.Dropbox(access_token)
    with open(local_file_path, "rb") as f:
        dbx.files_upload(f.read(), dropbox_file_path)
    print(f"File {local_file_path} uploaded to {dropbox_file_path}")


from dropbox.oauth import DropboxOAuth2FlowNoRedirect


def check_scopes():
    try:
        dbx = dropbox.Dropbox(access_token)
        account_info = dbx.users_get_current_account()
        print("Access token is valid for:", account_info.name.display_name)
        # Check available scopes
        response = dbx.auth_token_revoke()
        print("Token scopes:", response)
    except dropbox.exceptions.AuthError as err:
        print("Error in authorization:", err)


dropbox_file_path = "/your_file.txt"


def main():
    dropbox_file_path = "/thesis/filename.csv"
    local_file_path = "filename.csv"

    pd.DataFrame().to_csv(local_file_path)

    upload_to_dropbox(local_file_path, dropbox_file_path, access_token)


if __name__ == "__main__":
    main()
