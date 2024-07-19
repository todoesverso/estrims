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


def get_stream_bs_script(stream, is_video=False):
    channel_url = stream.channel_url
    if is_video:
        channel_url = stream.channel_url + "/videos"
    logger.warning(stream.channel_url)
    html = getHTMLdocument(channel_url)
    pattern = re.compile(r"ytInitialData = (.*);", re.MULTILINE | re.DOTALL)
    soup = BeautifulSoup(html, features="html.parser")
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
    if not keys:
        return None

    for path in keys:
        try:
            return access_path(data, path)
        except (KeyError, IndexError, TypeError):
            continue
    return None


def access_path(data, path):
    for key in path:
        if isinstance(data, dict):
            data = data[key]
        elif isinstance(data, list):
            data = data[key]
        else:
            raise TypeError(f"Expected dict or list, got {type(data).__name__}")
    return data


def get_title(base_path):
    paths = [["title", "runs", 0, "text"], ["title", "simpleText"]]
    return get_nested_value(base_path, paths)


def parse_latest_video(script_dict):
    logger.warning("LAST VIDEO")
    title = ""
    id = ""
    base_path_array = [
        "contents",
        "twoColumnBrowseResultsRenderer",
        "tabs",
        1,
        "tabRenderer",
        "content",
        "richGridRenderer",
        "contents",
        0,
        "richItemRenderer",
        "content",
        "videoRenderer",
    ]
    paths = [
        [
            *base_path_array,
        ],
    ]

    ret = get_nested_value(script_dict, paths)
    title = get_title(ret)
    id = ret["videoId"]

    ret_dict = {
        "last_video_title": title,
        "last_video_id": id,
    }
    logger.warning(ret_dict)

    return ret_dict


def parse_latest_stream(script_dict):
    logger.warning("LAST")
    title = ""
    id = ""
    base_path = script_dict["contents"]["twoColumnBrowseResultsRenderer"]["tabs"][0][
        "tabRenderer"
    ]["content"]["sectionListRenderer"]["contents"]
    base_path_array = [
        "contents",
        "twoColumnBrowseResultsRenderer",
        "tabs",
        0,
        "tabRenderer",
        "content",
        "sectionListRenderer",
        "contents",
    ]
    paths = [
        [
            *base_path_array,
            3,
            "itemSectionRenderer",
            "contents",
            0,
            "shelfRenderer",
            "content",
            "horizontalListRenderer",
            "items",
            0,
            "gridVideoRenderer",
        ],
        [
            *base_path_array,
            4,
            "itemSectionRenderer",
            "contents",
            0,
            "shelfRenderer",
            "content",
            "horizontalListRenderer",
            "items",
            0,
            "gridVideoRenderer",
        ],
        [
            *base_path_array,
            0,
            "itemSectionRenderer",
            "contents",
            0,
            "channelVideoPlayerRenderer",
        ],
        [
            *base_path_array,
            0,
            "itemSectionRenderer",
            "contents",
            0,
            "shelfRenderer",
            "content",
            "horizontalListRenderer",
            "items",
            0,
            "gridVideoRenderer",
        ],
        [
            *base_path_array,
            1,
            "itemSectionRenderer",
            "contents",
            0,
            "shelfRenderer",
            "content",
            "horizontalListRenderer",
            "items",
            0,
            "gridVideoRenderer",
        ],
    ]

    ret = get_nested_value(script_dict, paths)
    title = get_title(ret)
    logger.error(ret)
    id = ret["videoId"]

    ret_dict = {
        "last_video_title": title,
        "last_video_id": id,
    }
    logger.warning(ret_dict)

    return ret_dict


def new_stream_status(stream):
    script_dict = get_stream_bs_script(stream)
    script_dict_video = get_stream_bs_script(stream, is_video=True)
    if script_dict:
        # last = parse_latest_stream(script_dict)
        last = parse_latest_video(script_dict_video)
        live = parse_live_stream(script_dict)

        stream_status = StreamStatus(
            datetime=str(datetime.now()),
            stream=stream,
            stream_key=stream.title,
            **last,
            **live,
        )

        stream_status.create_or_update_to_db(
            asdict(stream_status), key={"stream_key": stream_status.stream.title}
        )


if __name__ == "__main__":
    streams_list = [
        Stream(
            title="Nico Guthmann",
            channel_url="https://www.youtube.com/@NicoGuthmann",
        ),
        Stream(
            title="BLENDER",
            channel_url="https://www.youtube.com/@estoesblender",
        ),
        Stream(
            title="Radio con vos",
            channel_url="https://www.youtube.com/@RadioConVos89.9",
        ),
        Stream(
            title="Gelatina",
            channel_url="https://www.youtube.com/@SomosGelatina",
        ),
        Stream(
            title="Futurock FM",
            channel_url="https://www.youtube.com/@futurock",
        ),
        Stream(
            title="posdata",
            channel_url="https://www.youtube.com/@Posdata_ar",
        ),
        Stream(
            title="Cenital",
            channel_url="https://www.youtube.com/@Cenitalcom",
        ),
        Stream(
            title="Factoria 1251",
            channel_url="https://www.youtube.com/@factoria1251",
        ),
        Stream(
            title="OLGA",
            channel_url="https://www.youtube.com/@olgaenvivo_",
        ),
        Stream(
            title="LUZU TV",
            channel_url="https://www.youtube.com/@luzutv",
        ),
        Stream(
            title="Picnoc Extraterrestre",
            channel_url="https://www.youtube.com/@Picnic.Extraterrestre",
        ),
        Stream(
            title="Pais de Boludos",
            channel_url="https://www.youtube.com/@PaisDeBoludos",
        ),
        Stream(
            title="Peroncho Delivery",
            channel_url="https://www.youtube.com/@PeronchoStandUp",
        ),
        Stream(
            title="Mate",
            channel_url="https://www.youtube.com/@somosmatear",
        ),
        Stream(
            title="El Destape",
            channel_url="https://www.youtube.com/@ElDestapeTV",
        ),
        Stream(
            title="Mano a Mano",
            channel_url="https://www.youtube.com/@ManoaMano-jz7up",
        ),
        Stream(
            title="220 Podcast",
            channel_url="https://www.youtube.com/@220Podcast",
        ),
        Stream(
            title="Eva TV",
            channel_url="https://www.youtube.com/@evaenvivo",
        ),
        Stream(
            title="Urbana Play",
            channel_url="https://www.youtube.com/@UrbanaPlayFM",
        ),
    ]

    streams = Streams(streams=streams_list)

    for s in streams.streams:
        new_stream_status(s)
