from dataclasses import dataclass, asdict
from typing import List
from datetime import datetime
from pysondb import db
from bs4 import BeautifulSoup
import requests
import re
import json


import logging

logger = logging.getLogger(__name__)


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
    last_video_id: str
    last_video_title: str

    @property
    def db(self):
        return db.getDb("data.json")


def getHTMLdocument(url):
    response = requests.get(url)
    return response.text


def save_streams_to_db(streams: Streams):
    for s in streams.streams:
        s.write_to_db_if_not_exists(asdict(s), key={"title": s.title})


def get_stream_bs_script(stream):
    channel_url = stream.channel_url
    logger.warning(stream.channel_url)
    html = getHTMLdocument(channel_url)
    pattern = re.compile(r"ytInitialData = (.*);", re.MULTILINE | re.DOTALL)
    soup = BeautifulSoup(html, features="lxml")
    script = soup.find("script", string=pattern)
    data = None
    if script:
        match = pattern.search(script.text)
        data = json.loads(match.group(1))

    return data


def parse_live_stream(script_dict):
    logger.warning("LIVE")
    try:
        base_path = script_dict["contents"]["twoColumnBrowseResultsRenderer"]["tabs"][
            0
        ]["tabRenderer"]["content"]["sectionListRenderer"]["contents"][0][
            "itemSectionRenderer"
        ][
            "contents"
        ][
            0
        ][
            "channelFeaturedContentRenderer"
        ][
            "items"
        ][
            0
        ][
            "videoRenderer"
        ]

        return {
            "live_title": base_path["title"]["runs"][0]["text"],
            "live_id": base_path["videoId"],
        }
    except KeyError:
        return {"live_title": None, "live_id": None}


def get_nested_value(data, keys):
    """Recursively try to get a nested value in a JSON-like dictionary.

    Args:
        data (dict): The JSON data as a dictionary.
        keys (list): A list of lists, where each inner list is a possible path of keys to the target value.

    Returns:
        The value if found, otherwise None.
    """
    if not keys:
        return None

    for path in keys:
        try:
            return access_path(data, path)
        except KeyError:
            continue
    return None


def access_path(data, path):
    """Access a value in a nested dictionary following a specific path.

    Args:
        data (dict): The JSON data as a dictionary.
        path (list): A list of keys representing the path to the target value.

    Returns:
        The value found at the end of the path.

    Raises:
        KeyError: If the path is not found in the data.
    """
    for key in path:
        data = data[key]
    return data


def parse_latest_stream(script_dict):
    logger.warning("LAST")
    title = ""
    id = ""
    base_path = script_dict["contents"]["twoColumnBrowseResultsRenderer"]["tabs"][0][
        "tabRenderer"
    ]["content"]["sectionListRenderer"]["contents"]
    __paths = [["content", "twoColumnBrowseResultsRenderer"]]
    try:
        ret = base_path[4]["itemSectionRenderer"]["contents"][0]["shelfRenderer"][
            "content"
        ]["horizontalListRenderer"]["items"][0]["gridVideoRenderer"]
        title = ret["title"]["runs"][0]["text"]
        id = ret["videoId"]
    except (KeyError, IndexError):
        pass

    try:
        ret = base_path[0]["itemSectionRenderer"]["contents"][0][
            "channelVideoPlayerRenderer"
        ]
        title = ret["title"]["runs"][0]["text"]
        id = ret["videoId"]
    except KeyError:
        ret = base_path[1]["itemSectionRenderer"]["contents"][0]["shelfRenderer"][
            "content"
        ]["horizontalListRenderer"]["items"][0]["gridVideoRenderer"]

        title = ret["title"]["simpleText"]
        id = ret["videoId"]
    return {
        "last_video_title": title,
        "last_video_id": id,
    }


def new_stream_status(stream):
    script_dict = get_stream_bs_script(stream)
    if script_dict:
        last = parse_latest_stream(script_dict)
        live = parse_live_stream(script_dict)

        stream_status = StreamStatus(
            datetime=str(datetime.now()),
            stream=stream,
            stream_key=stream.title,
            **last,
            **live
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
