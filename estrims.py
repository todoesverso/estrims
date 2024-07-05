import os
from dataclasses import dataclass, asdict
from typing import List
from datetime import datetime
from pysondb import db
import googleapiclient.discovery
from bs4 import BeautifulSoup
import requests
import re
import json


import logging

logger = logging.getLogger(__name__)
API_KEY = os.environ.get("API_KEY")


class BaseDb:
    @property
    def db(self):
        return db.getDb(str(self.__class__.__name__).lower() + ".json")

    def write_to_db(self, data):
        self.db.add(data)

    def write_to_db_if_not_exists(self, data, key: dict):
        exist = self.db.getByQuery(key)
        if exist:
            logger.warning("Key %s already exists", key)
            return False
        self.write_to_db(data)

    def create_or_update_to_db(self, data, key: dict):
        exist = self.db.getByQuery(key)
        if exist:
            logger.warning("Key %s already exists. Updating", key)
            self.db.updateByQuery(key, data)
        else:
            self.write_to_db(data)


@dataclass
class Stream(BaseDb):
    title: str
    channel_id: str
    channel_url: str


@dataclass
class Streams:
    streams: List[Stream]


@dataclass
class StreamStatus(BaseDb):
    datetime: datetime | str
    stream: Stream
    stream_key: str  # use stream.title
    live_id: str | None
    live_title: str | None
    live_description: str | None
    live_thumbnail: str | None
    last_video_id: str
    last_video_title: str
    last_video_description: str
    last_video_thumbnail: str

    @property
    def db(self):
        return db.getDb("data.json")


def getHTMLdocument(url):
    response = requests.get(url)
    return response.text


def save_streams_to_db(streams: Streams):
    for s in streams.streams:
        s.write_to_db_if_not_exists(asdict(s), key={"title": s.title})


def get_live_stream(channel_id):
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=API_KEY)

    request = youtube.search().list(
        part="snippet", channelId=channel_id, eventType="live", type="video"
    )
    response = request.execute()

    live_streams = {
        "title": None,
        "videoId": None,
        "description": None,
        "thumbnail": None,
    }
    if response.get("items"):
        item = response["items"][0]
        live_streams = {
            "title": item["snippet"]["title"],
            "videoId": item["id"]["videoId"],
            "description": item["snippet"]["description"],
            "thumbnail": item["snippet"]["thumbnails"]["default"]["url"],
        }

    return live_streams


def get_live_stream_bs(stream):
    channel_url = stream.channel_url
    logger.warning("Parsing LIVE: %s", stream.channel_url)

    live_streams = {
        "title": None,
        "videoId": None,
        "description": None,
        "thumbnail": None,
    }

    html = getHTMLdocument(channel_url)
    pattern = re.compile(r"ytInitialData = (.*);", re.MULTILINE | re.DOTALL)
    soup = BeautifulSoup(html)
    script = soup.find("script", text=pattern)
    if script:
        match = pattern.search(script.text)
        data = json.loads(match.group(1))

        base_path = data["contents"]["twoColumnBrowseResultsRenderer"]["tabs"][0][
            "tabRenderer"
        ]["content"]["sectionListRenderer"]["contents"][0]["itemSectionRenderer"][
            "contents"
        ][
            0
        ]
        try:
            live = base_path["channelFeaturedContentRenderer"]["items"][0][
                "videoRenderer"
            ]
            live_streams = {
                "title": live["title"]["runs"][0]["text"],
                "videoId": live["videoId"],
                "description": "",
                "thumbnail": "",
            }

        except Exception:
            pass

    return live_streams


def get_last_stream_bs(stream):
    channel_url = stream.channel_url
    logger.warning("Parsing LAST: %s", stream.channel_url)

    last_video = {
        "title": "",
        "videoId": "",
        "description": "",
        "thumbnail": "",
    }

    html = getHTMLdocument(channel_url + "/videos")
    pattern = re.compile(r"ytInitialData = (.*);", re.MULTILINE | re.DOTALL)
    soup = BeautifulSoup(html)
    script = soup.find("script", text=pattern)
    if script:
        match = pattern.search(script.text)
        data = json.loads(match.group(1))

        last = data["contents"]["twoColumnBrowseResultsRenderer"]["tabs"][1][
            "tabRenderer"
        ]["content"]["richGridRenderer"]["contents"][0]["richItemRenderer"]["content"][
            "videoRenderer"
        ]
        last_video = {
                "title": last["title"]["runs"][0]["text"],
                "videoId": last["videoId"],
                "description": "",
                "thumbnail": "",
            }

    return last_video


def get_latest_video(channel_id):
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=API_KEY)

    request = youtube.search().list(
        part="snippet",
        channelId=channel_id,
        order="date",
        maxResults=1,
        type="video",
    )
    response = request.execute()

    latest_video = None
    if response.get("items"):
        item = response["items"][0]
        latest_video = {
            "title": item["snippet"]["title"],
            "videoId": item["id"]["videoId"],
            "description": item["snippet"]["description"],
            "publishedAt": item["snippet"]["publishedAt"],
            "thumbnail": item["snippet"]["thumbnails"]["default"]["url"],
        }

    return latest_video


def new_stream_status(stream):
    # last = get_latest_video(stream.channel_id)
    # live = get_live_stream(stream.channel_id)

    last = get_last_stream_bs(stream)
    live = get_live_stream_bs(stream)

    stream_status = StreamStatus(
        datetime=str(datetime.now()),
        stream=stream,
        stream_key=stream.title,
        live_id=live["videoId"],
        live_description=live["description"],
        live_title=live["title"],
        live_thumbnail=live["thumbnail"],
        last_video_id=last["videoId"],
        last_video_title=last["title"],
        last_video_thumbnail=last["thumbnail"],
        last_video_description=last["description"],
    )

    stream_status.create_or_update_to_db(
        asdict(stream_status), key={"stream_key": stream_status.stream.title}
    )


if __name__ == "__main__":
    streams_list = [
        Stream(
            title="Nico Guthmann",
            channel_id="UCpvYnSZNyBxF2MSWvju06jg",
            channel_url="https://www.youtube.com/@NicoGuthmann",
        ),
        Stream(
            title="BLENDER",
            channel_id="UC6pJGaMdx5Ter_8zYbLoRgA",
            channel_url="https://www.youtube.com/@estoesblender",
        ),
        Stream(
            title="Radio con vos",
            channel_id="UCxDteokWBemJvLI_I0VUGdA",
            channel_url="https://www.youtube.com/@RadioConVos89.9",
        ),
        Stream(
            title="Gelatina",
            channel_id="UCWSfXECGo1qK_H7SXRaUSMg",
            channel_url="https://www.youtube.com/@SomosGelatina",
        ),
        Stream(
            title="Futurock FM",
            channel_id="",
            channel_url="https://www.youtube.com/@futurock",
        ),
        Stream(
            title="posdata",
            channel_id="",
            channel_url="https://www.youtube.com/@Posdata_ar",
        ),
        Stream(
            title="Cenital",
            channel_id="",
            channel_url="https://www.youtube.com/@Cenitalcom",
        ),
        Stream(
            title="Factoria 1251",
            channel_id="",
            channel_url="https://www.youtube.com/@factoria1251",
        ),
        Stream(
            title="OLGA",
            channel_id="",
            channel_url="https://www.youtube.com/@olgaenvivo_",
        ),
        Stream(
            title="LUZU TV",
            channel_id="",
            channel_url="https://www.youtube.com/@luzutv",
        ),
    ]

    streams = Streams(streams=streams_list)
    # save_streams_to_db(streams)

    for s in streams.streams:
        new_stream_status(s)
