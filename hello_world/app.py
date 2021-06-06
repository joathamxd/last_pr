from json import loads, dumps
from hello_world.tweet_evaluator.error import ERROR_READING_TWEET
from datetime import datetime
from pandas import DataFrame
from tweet_evaluator import Process
from pickle import load


# import requests
TEAM_NAME = "Los merequetengues"


def get_origin_data(body):
    try:
        tweet = body["tweet_text"]
        assert isinstance(tweet, str)
        return tweet
    except Exception as e:
        return ERROR_READING_TWEET


def tweet_to_df(tweet):
    return DataFrame([tweet, ],
                     columns=['tweet_text'])


def evaluate_trained_model(data):
    model = load(open("../hello_world/data/models/los_merequetengues.pk", 'rb'))
    predict = model.predict(data)
    prob = model.predict_proba(data)
    return {
        "proba_mixed": prob[0][0],
        "proba_negative": prob[0][1],
        "proba_neutral": prob[0][2],
        "proba_positive": prob[0][3],
        "class": predict[0],
    }


def lambda_request(probs, cluster="Random cluster"):
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
                    "class": probs["class"],
                    "cluster": cluster,
                },
            }
        ),
    }


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
    data = tweet_to_df(tweet)
    process = Process(data)
    process.run()
    features = [
        'tweet_mensaje',
        'n_emojis', 'n_lower', 'n_upper',
        'n_digit', 'n_whitespaces', 'n_words', 'has_tags', 'has_hashtag',
        'has_urls', 'n_exclamation', 'n_question', 'n_hashtag', 'n_tags',
        'n_urls', 'count_personal_positive',
        'count_personal_negative'
    ]
    prob = evaluate_trained_model(process.data[features])

    return lambda_request(prob, cluster=tweet)


