from json import loads, dumps
from tweet_evaluator.error import ERROR_READING_TWEET
from datetime import datetime
from pandas import DataFrame
from tweet_evaluator import Process
from pickle import load
from numpy import around


# import requests
TEAM_NAME = "Los merequetengues"


def get_origin_data(body):
    print("Reading tweet from request")
    try:
        tweet = body["tweet_text"]
        assert isinstance(tweet, str)
        return tweet
    except Exception as e:
        return ERROR_READING_TWEET


def tweet_to_df(tweet):
    print("Converting Tweet to DF")
    return DataFrame([tweet, ],
                     columns=['tweet_text'])


def evaluate_trained_model(data):
    print("Reading pickle model and evaluating")
    min_max = load(open('data/models/los_merequetengues_mixman.pk', 'rb'))
    model = load(open("data/models/los_merequetengues.pk", 'rb'))
    model_cluster = load(open("data/models/los_merequetenguesC.pk", 'rb'))

    scaled = min_max.transform(data)
    predict = model.predict(scaled)
    prob = model.predict_proba(scaled)
    cluster = model_cluster.predict(scaled)

    return {
        "proba_mixed": prob[0][0],
        "proba_negative": prob[0][1],
        "proba_neutral": prob[0][2],
        "proba_positive": prob[0][3],
        "class": predict[0],
        "cluster": convert_cluster(cluster[0])
    }


def convert_cluster(cluster):
    if cluster == 0:
        return "Los apasionados"
    if cluster == 1:
        return "Los que Responden"
    if cluster == 2:
        return "Las Reacciones"
    if cluster == 3:
        return "Los Analistas"


def lambda_request(probs):
    print("Requesting data")
    today = datetime.today().strftime('%d/%m/%YT%H:%M:%S')
    return {
        "statusCode": 200,
        "body": dumps(
            {
                "message": "Tweet qualified.",
                "data": {
                    "timestamp": str(today),
                    "team_name": TEAM_NAME,
                    "proba_positive": around(probs["proba_positive"], decimals=10),
                    "proba_negative": around(probs["proba_negative"], decimals=10),
                    "proba_neutral": around(probs["proba_neutral"], decimals=10),
                    "proba_mixed": around(probs["proba_mixed"], decimals=10),
                    "class": probs["class"],
                    "cluster": str(probs["cluster"]),
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
    process = Process(data, "")

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

    return lambda_request(prob)


