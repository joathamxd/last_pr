from json import loads, dumps
from error import ERROR_READING_TWEET, ERROR_TRANSLATING_TWEET
from datetime import datetime
from googletrans import Translator

# import requests
TEAM_NAME = "Los merequetengues"


def get_origin_data(body):
    try:
        tweet = body["tweet_text"]
        assert isinstance(tweet, str)
        return tweet
    except Exception as e:
        return ERROR_READING_TWEET


def lambda_request(probs, tweet_class="POSITIVE", cluster="Random cluster"):
    today = datetime.today().strftime('%d/%m/%YT%H:%M:%S')
    return {
        "statusCode": 200,
        "body": dumps(
            {
                "message": "Tweet qualified.",
                "data": {
                    "timestamp": str(today),
                    "team_name": TEAM_NAME,
                    "proba_positive": float(probs["proba_positive"]),
                    "proba_negative": float(probs["proba_negative"]),
                    "proba_neutral": float(probs["proba_neutral"]),
                    "proba_mixed": float(probs["proba_mixed"]),
                    "class": tweet_class,
                    "cluster": cluster,
                },
            }
        ),
    }


def translate_tweet(tweet):
    try:
        translator = Translator()
        tr = translator.translate(tweet, src="es", dest="en")
        return tr
    except Exception as e:
        return ERROR_TRANSLATING_TWEET


def lambda_handler(event, context):
    """Sample pure Lambda function

    Parameters
    ----------
    event: dict, required
        API Gateway Lambda Proxy Input Format

        Event doc: https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html#api-gateway-simple-proxy-for-lambda-input-format

    context: object, required
        Lambda Context runtime methods and attributes

        Context doc: https://docs.aws.amazon.com/lambda/latest/dg/python-context-object.html

    Returns
    ------
    API Gateway Lambda Proxy Output Format: dict

        Return doc: https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html
    """

    body = loads(event["body"])
    tweet = get_origin_data(body)
    tweet_es = translate_tweet(tweet)


    test_prob = {
        "proba_positive": 0.056,
        "proba_negative": 0.532,
        "proba_neutral": 0.102,
        "proba_mixed": 0.333,
    }

    return lambda_request(test_prob, cluster=tweet_es)


