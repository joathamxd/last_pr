from json import dumps

ERROR_READING_TWEET = {
    "statusCode": 500,
    "body": dumps(
        {
            "message": "Please, provide a valid Tweet.",
            "data": "Error reading tweet."
        }
    )
}


ERROR_TRANSLATING_TWEET = "Error translating Tweet."
