import requests


# Save chunk of streaming response
def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    i = 0
    length = len(response.content)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

            i = i + 1

    print("DOWNLOAD FILE SUCCESS")


# Download file
def download_file(url, save_path):
    session = requests.Session()

    response = session.get(url, stream=True)

    save_response_content(response, save_path)


# Download google drive file
def download_drive_file(id, save_path):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    url = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(url, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(url, params=params, stream=True)

    save_response_content(response, save_path)

