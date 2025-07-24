from dataclasses import dataclass, asdict
from typing import List
from datetime import datetime
from pysondb import db
from bs4 import BeautifulSoup
import requests
import re
import json
import logging
import duckdb as dd
import sys


con = dd.connect("views.db")
logger = logging.getLogger(__name__)


class ViewsDB:
    DB_NAME = "views.db"

    def __init__(self) -> None:
        self.con = dd.connect(self.DB_NAME)
        self._create_view_table()

    def _create_view_table(self) -> None:
        self.con.execute("""
        CREATE TABLE IF NOT EXISTS views (
            channel_id VARCHAR NOT NULL,
            channel_name VARCHAR,
            live_timestamp TIMESTAMP NOT NULL,
            viewers_count INTEGER NOT NULL,
            PRIMARY KEY (channel_id, live_timestamp)
        );
        """)

    def insert(self, channel_id, channel_name, viewers_count) -> None:
        self.con.execute(
            "INSERT INTO views VALUES (? ,?, ?, ?)",
            (channel_id, channel_name, datetime.now(), viewers_count),
        )


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
    thumbnail: str
    live_id: str | None
    live_title: str | None
    viewing: int

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
    logger.warning(channel_url)
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
        ]["contents"][0]["channelFeaturedContentRenderer"]["items"][0]["videoRenderer"]

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


def parse_thumbnail(script_dict):
    return script_dict["metadata"]["channelMetadataRenderer"]["avatar"]["thumbnails"][
        0
    ]["url"]


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


def parse_curr_view(script_dict):
    ret = 0
    try:
        cv = script_dict["contents"]["twoColumnBrowseResultsRenderer"]["tabs"][0][
            "tabRenderer"
        ]["content"]["sectionListRenderer"]["contents"][0]["itemSectionRenderer"][
            "contents"
        ][0]["channelFeaturedContentRenderer"]["items"][0]["videoRenderer"][
            "viewCountText"
        ]["runs"][0]["text"]
        ret = int(cv.replace(",", ""))
    except Exception as e:
        pass

    logger.error(ret)
    return ret


def new_stream_status(stream):
    script_dict = get_stream_bs_script(stream)
    if script_dict:
        thumbnail = ""
        live = {"live_id": None, "live_title": None}
        viewing = 0

        try:
            thumbnail = parse_thumbnail(script_dict)
            live = parse_live_stream(script_dict)
            viewing = parse_curr_view(script_dict)
        except Exception:
            logger.warning("Failed to parse %s", stream)

        stream_status = StreamStatus(
            datetime=str(datetime.now()),
            stream=stream,
            stream_key=stream.title,
            thumbnail=thumbnail,
            viewing=viewing,
            **live,
        )

        stream_status.create_or_update_to_db(
            asdict(stream_status), key={"stream_key": stream_status.stream.title}
        )
        return stream_status


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
        Stream(title="Bondi Live", channel_url="https://www.youtube.com/@Bondi_liveok"),
        Stream(
            title="Vorterix", channel_url="https://www.youtube.com/@VorterixOficial"
        ),
        Stream(title="Neura Media", channel_url="https://www.youtube.com/@NeuraMedia"),
        Stream(title="RepublicaZ", channel_url="https://www.youtube.com/@republicaz"),
        Stream(
            title="La Casa Stream", channel_url="https://www.youtube.com/@somoslacasa"
        ),
        Stream(title="Ahora Play", channel_url="https://www.youtube.com/@tesla1923"),
        Stream(title="Mix On", channel_url="https://www.youtube.com/@mixontv_"),
        Stream(title="Clank!", channel_url="https://www.youtube.com/@clank_media"),
        Stream(title="YEITE", channel_url="https://www.youtube.com/@somosyeite"),
        Stream(title="Chingon", channel_url="https://www.youtube.com/@chingonenvivo"),
        Stream(title="CEIBO", channel_url="https://www.youtube.com/@CEIBOARGENTINA"),
        Stream(title="Brindis TV", channel_url="https://www.youtube.com/@brindistv"),
        Stream(title="Laca Stream", channel_url="https://www.youtube.com/@lacastream"),
        Stream(
            title="Norita Stream",
            channel_url="https://www.youtube.com/@NoritaStreaming",
        ),
        Stream(title="Ziesta TV", channel_url="https://www.youtube.com/@ZiestaTV"),
        Stream(title="MOSTRI TV", channel_url="https://www.youtube.com/@mostritv_"),
        Stream(
            title="A la Estratosfera",
            channel_url="https://www.youtube.com/@estratosferaok",
        ),
        Stream(
            title="Polenta para revolver",
            channel_url="https://www.youtube.com/@polentapararevolver",
        ),
        Stream(title="Re FM 107.3", channel_url="https://www.youtube.com/@ReFM107.3"),
        Stream(title="Chimi Canal", channel_url="https://www.youtube.com/@ChimiCanal"),
        Stream(
            title="Diario Alfil Cordoba",
            channel_url="https://www.youtube.com/@diario.alfil.cordoba",
        ),
        Stream(
            title="Nada del otro mundo",
            channel_url="https://www.youtube.com/@NadadelOtroMundo2024",
        ),
        Stream(
            title="BorderPeriodismo",
            channel_url="https://www.youtube.com/@border.periodismo",
        ),
        Stream(title="Telefe", channel_url="https://www.youtube.com/@Telefe"),
        Stream(
            title="Bunker", channel_url="https://www.youtube.com/@bunkeraguantadero"
        ),
        Stream(title="Radio TU", channel_url="https://www.youtube.com/@radiotu"),
        Stream(title="SODA", channel_url="https://www.youtube.com/@quierosoda"),
        Stream(
            title="AZZ Contenidos", channel_url="https://www.youtube.com/@AZZContenidos"
        ),
        Stream(title="Data Diario", channel_url="https://www.youtube.com/@datadiario"),
        Stream(title="PelaVision", channel_url="https://www.youtube.com/@pelavision-"),
        Stream(
            title="Radio Kamikaze",
            channel_url="https://www.youtube.com/@RadioKamikazeok",
        ),
        Stream(
            title="Che Corrientes",
            channel_url="https://www.youtube.com/@checorrientesstreaming",
        ),
        Stream(title="Crudo TV", channel_url="https://www.youtube.com/@SomosCrudoTV"),
        Stream(title="Mistica TV", channel_url="https://www.youtube.com/@misticatv."),
        Stream(
            title="La Casa del Stream",
            channel_url="https://www.youtube.com/@lacasadelstreaming",
        ),
        Stream(title="NAMIK TV", channel_url="https://www.youtube.com/@namiktv"),
        Stream(title="FAMATV", channel_url="https://www.youtube.com/@famatvok"),
        Stream(
            title="CitricaRadio", channel_url="https://www.youtube.com/@SomosCitrica"
        ),
        Stream(title="DIGI TV", channel_url="https://www.youtube.com/@DIGITV_"),
        Stream(title="Vermut", channel_url="https://www.youtube.com/@somosvermut"),
        Stream(title="DGO", channel_url="https://www.youtube.com/@dgo_latam"),
        Stream(
            title="Canal 10 Cordoba",
            channel_url="https://www.youtube.com/@canal10cordoba",
        ),
        Stream(
            title="Ivana Szerman", channel_url="https://www.youtube.com/@ivanaszerman"
        ),
        Stream(title="AZZ", channel_url="https://www.youtube.com/@somosazz"),
        Stream(title="Abitare", channel_url="https://www.youtube.com/@Somosabitare"),
        Stream(
            title="Carnaval Stream",
            channel_url="https://www.youtube.com/@CarnavalStream",
        ),
    ]

    streams = Streams(streams=streams_list)

    # vdb = ViewsDB()
    for s in streams.streams:
        status = new_stream_status(s)
        # if status is not None:
        #    vdb.insert(status.stream.title, status.stream.channel_url, status.viewing)
