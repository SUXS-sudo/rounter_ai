from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from core.poi_artifact_store import MANIFEST_NAME
from models.config import settings


STATIC_ZONE_METADATA: dict[str, dict[str, dict[str, Any]]] = {
    "成都": {
        "春熙路商圈": {"center": (104.081, 30.657), "district": "锦江区"},
        "太古里商圈": {"center": (104.084, 30.654), "district": "锦江区"},
        "宽窄巷子商圈": {"center": (104.060, 30.670), "district": "青羊区"},
        "九眼桥商圈": {"center": (104.092, 30.641), "district": "锦江区"},
        "科华路商圈": {"center": (104.068, 30.618), "district": "武侯区"},
        "建设路商圈": {"center": (104.105, 30.665), "district": "成华区"},
        "万象城商圈": {"center": (104.098, 30.632), "district": "锦江区"},
        "天府新区商圈": {"center": (104.063, 30.500), "district": "天府新区"},
    },
    "上海": {
        "新天地商圈": {"center": (121.475, 31.216), "district": "黄浦区"},
        "南京路商圈": {"center": (121.475, 31.235), "district": "黄浦区"},
        "陆家嘴商圈": {"center": (121.499, 31.239), "district": "浦东新区"},
        "徐家汇商圈": {"center": (121.438, 31.188), "district": "徐汇区"},
        "静安寺商圈": {"center": (121.448, 31.223), "district": "静安区"},
    },
    "北京": {
        "三里屯商圈": {"center": (116.455, 39.933), "district": "朝阳区"},
        "王府井商圈": {"center": (116.414, 39.914), "district": "东城区"},
        "中关村商圈": {"center": (116.316, 39.982), "district": "海淀区"},
        "西单商圈": {"center": (116.374, 39.910), "district": "西城区"},
        "国贸商圈": {"center": (116.462, 39.909), "district": "朝阳区"},
    },
    "厦门": {
        "中山路商圈": {"center": (118.075, 24.450), "district": "思明区"},
        "鼓浪屿": {"center": (118.070, 24.445), "district": "思明区"},
        "SM商圈": {"center": (118.118, 24.479), "district": "湖里区"},
        "曾厝垵商圈": {"center": (118.098, 24.440), "district": "思明区"},
    },
}


STATIC_ZONE_ALIASES: dict[str, str] = {
    "春熙路": "春熙路商圈",
    "太古里": "太古里商圈",
    "宽窄巷子": "宽窄巷子商圈",
    "九眼桥": "九眼桥商圈",
    "万象城": "万象城商圈",
    "建设路": "建设路商圈",
    "科华路": "科华路商圈",
    "天府新区": "天府新区商圈",
    "王府井": "王府井商圈",
    "三里屯": "三里屯商圈",
    "西单": "西单商圈",
    "国贸": "国贸商圈",
    "中关村": "中关村商圈",
    "五道口": "五道口商圈",
    "望京": "望京商圈",
    "前门": "前门商圈",
    "南京路": "南京路商圈",
    "淮海路": "淮海路商圈",
    "陆家嘴": "陆家嘴商圈",
    "徐家汇": "徐家汇商圈",
    "静安寺": "静安寺商圈",
    "五角场": "五角场商圈",
    "新天地": "新天地商圈",
    "天河城": "天河城商圈",
    "北京路": "北京路商圈",
    "珠江新城": "珠江新城商圈",
    "上下九": "上下九商圈",
    "华强北": "华强北商圈",
    "东门": "东门商圈",
    "海岸城": "海岸城商圈",
    "西湖": "西湖商圈",
    "武林广场": "武林广场商圈",
    "湖滨": "湖滨商圈",
    "钱江新城": "钱江新城商圈",
    "江汉路": "江汉路商圈",
    "光谷": "光谷商圈",
    "楚河汉街": "楚河汉街商圈",
    "钟楼": "钟楼商圈",
    "小寨": "小寨商圈",
    "大雁塔": "大雁塔商圈",
    "回民街": "回民街商圈",
    "解放碑": "解放碑商圈",
    "观音桥": "观音桥商圈",
    "洪崖洞": "洪崖洞商圈",
    "新街口": "新街口商圈",
    "夫子庙": "夫子庙商圈",
    "滨江道": "滨江道商圈",
    "观前街": "观前街商圈",
    "金鸡湖": "金鸡湖商圈",
    "五一广场": "五一广场商圈",
    "太平街": "太平街商圈",
    "台东": "台东商圈",
    "二七广场": "二七广场商圈",
    "中山路": "中山路商圈",
    "曾厝垵": "曾厝垵商圈",
    "鼓浪屿": "鼓浪屿",
    "SM": "SM商圈",
    "南屏街": "南屏街商圈",
    "翠湖": "翠湖商圈",
    "青泥洼桥": "青泥洼桥商圈",
    "星海广场": "星海广场商圈",
    "三亚湾": "三亚湾商圈",
    "亚龙湾": "亚龙湾商圈",
    "海棠湾": "海棠湾商圈",
    "大东海": "大东海商圈",
    "大研古城": "大研古城",
    "束河古镇": "束河古镇",
}


_zone_catalog_cache: dict[str, Any] | None = None



def _manifest_path() -> Path:
    path = Path(settings.pois_file)
    if path.is_dir():
        return path / MANIFEST_NAME
    if path.name == MANIFEST_NAME:
        return path
    return Path("")



def _sqlite_path() -> Path:
    path = Path(settings.pois_file)
    if path.suffix.lower() in {'.db', '.sqlite', '.sqlite3'}:
        return path
    return Path("")



def load_zone_catalog() -> dict[str, Any]:
    global _zone_catalog_cache
    if _zone_catalog_cache is not None:
        return _zone_catalog_cache

    metadata = {city: {zone: dict(info) for zone, info in zones.items()} for city, zones in STATIC_ZONE_METADATA.items()}
    aliases = dict(STATIC_ZONE_ALIASES)

    sqlite_path = _sqlite_path()
    if sqlite_path and sqlite_path.exists():
        with sqlite3.connect(sqlite_path) as conn:
            rows = conn.execute(
                "SELECT city, zone, district, center_lng, center_lat, shard_aliases_json FROM zones ORDER BY city, zone"
            ).fetchall()
        for city, zone_name, district, center_lng, center_lat, shard_aliases_json in rows:
            city_zones = metadata.setdefault(str(city), {})
            zone_info = city_zones.setdefault(str(zone_name), {})
            if center_lng is not None and center_lat is not None:
                zone_info['center'] = (float(center_lng), float(center_lat))
            zone_info.setdefault('district', str(district or ''))
            aliases[str(zone_name)] = str(zone_name)
            if str(zone_name).endswith('商圈'):
                aliases.setdefault(str(zone_name)[:-2], str(zone_name))
            aliases.setdefault(str(zone_name).replace('商圈', ''), str(zone_name))
            if shard_aliases_json:
                for alias in json.loads(shard_aliases_json):
                    alias_text = str(alias).strip()
                    if alias_text:
                        aliases.setdefault(alias_text, str(zone_name))
    else:
        manifest = _manifest_path()
        if manifest and manifest.exists():
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            for city, info in payload.get("cities", {}).items():
                city_zones = metadata.setdefault(city, {})
                districts = info.get("districts", {})
                for district_name in districts:
                    metadata.setdefault(city, {})
                for zone_name, zone_payload in info.get("zones", {}).items():
                    zone_info = city_zones.setdefault(zone_name, {})
                    if isinstance(zone_payload, str):
                        zone_payload = {'file': zone_payload}
                    center = zone_payload.get('center', [])
                    if len(center) == 2:
                        zone_info['center'] = tuple(center)
                    zone_info.setdefault("district", zone_payload.get('district') or _guess_zone_district(city, zone_name, info))
                    zone_info.setdefault('file', zone_payload.get('file', ''))
                    if zone_name not in aliases:
                        aliases[zone_name] = zone_name
                    if zone_name.endswith("商圈"):
                        aliases.setdefault(zone_name[:-2], zone_name)
                    aliases.setdefault(zone_name.replace("商圈", ""), zone_name)
    _zone_catalog_cache = {"metadata": metadata, "aliases": aliases}
    return _zone_catalog_cache



def _guess_zone_district(city: str, zone_name: str, city_info: dict[str, Any]) -> str:
    static = STATIC_ZONE_METADATA.get(city, {}).get(zone_name, {})
    district = str(static.get("district") or "")
    if district:
        return district
    districts = city_info.get("districts", {})
    if len(districts) == 1:
        return next(iter(districts))
    return ""



def get_zone_metadata() -> dict[str, dict[str, dict[str, Any]]]:
    return load_zone_catalog()["metadata"]



def get_zone_aliases() -> dict[str, str]:
    return load_zone_catalog()["aliases"]
