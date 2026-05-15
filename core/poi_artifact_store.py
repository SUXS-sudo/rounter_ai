from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PARTITIONED_FORMAT = "partitioned_json_v1"
MANIFEST_NAME = "manifest.json"
CITIES_DIR = "cities"
ALL_FILE = "all.json"
ZONES_DIR = "zones"
DISTRICTS_DIR = "districts"


@dataclass(frozen=True)
class PartitionedCityInfo:
    count: int
    all_file: str
    zones: dict[str, dict[str, Any]]
    districts: dict[str, str]


class PoiStore(Mapping[str, list[dict[str, Any]]]):
    def city_counts(self) -> dict[str, int]:
        raise NotImplementedError

    def load_city(self, city: str) -> list[dict[str, Any]]:
        raise NotImplementedError

    def load_scope(self, city: str, scope_key: str) -> list[dict[str, Any]] | None:
        return None

    def load_all_by_city(self) -> dict[str, list[dict[str, Any]]]:
        return {city: self.load_city(city) for city in self.city_counts()}

    def __getitem__(self, key: str) -> list[dict[str, Any]]:
        return self.load_city(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.city_counts())

    def __len__(self) -> int:
        return len(self.city_counts())

    def items(self):
        for city in self.city_counts():
            yield city, self.load_city(city)

    def get(self, key: str, default: Any = None):
        if key in self.city_counts():
            return self.load_city(key)
        return default


class MonolithicJsonPoiStore(PoiStore):
    def __init__(self, path: Path):
        self.path = path
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, dict):
            raise ValueError("Monolithic POI artifact must contain a JSON object keyed by city")
        self._data = {str(city): list(pois) for city, pois in payload.items()}
        self._counts = {city: len(pois) for city, pois in self._data.items()}

    def city_counts(self) -> dict[str, int]:
        return dict(self._counts)

    def load_city(self, city: str) -> list[dict[str, Any]]:
        return list(self._data.get(city, []))

    def load_all_by_city(self) -> dict[str, list[dict[str, Any]]]:
        return {city: list(pois) for city, pois in self._data.items()}


class PartitionedJsonPoiStore(PoiStore):
    def __init__(self, source: Path):
        manifest_path = source / MANIFEST_NAME if source.is_dir() else source
        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
        if manifest.get("format") != PARTITIONED_FORMAT:
            raise ValueError(f"Unsupported POI manifest format: {manifest.get('format')}")
        self.root = manifest_path.parent
        self._city_info = {
            str(city): PartitionedCityInfo(
                count=int(info.get("count", 0)),
                all_file=str(info.get("all_file", "")),
                zones={
                    str(zone): ({'file': str(path)} if isinstance(path, str) else dict(path))
                    for zone, path in info.get("zones", {}).items()
                },
                districts={
                    str(district): (str(path) if isinstance(path, str) else str(path.get('file', '')))
                    for district, path in info.get("districts", {}).items()
                },
            )
            for city, info in manifest.get("cities", {}).items()
        }
        self._city_cache: dict[str, list[dict[str, Any]]] = {}
        self._scope_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}


class SqlitePoiStore(PoiStore):
    def __init__(self, path: Path):
        self.path = path
        self._city_cache: dict[str, list[dict[str, Any]]] = {}
        self._scope_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self._city_counts: dict[str, int] | None = None

    def city_counts(self) -> dict[str, int]:
        if self._city_counts is None:
            with self._connect() as conn:
                rows = conn.execute("SELECT city, poi_count FROM cities ORDER BY city").fetchall()
            self._city_counts = {str(row[0]): int(row[1]) for row in rows}
        return dict(self._city_counts)

    def load_city(self, city: str) -> list[dict[str, Any]]:
        if city in self._city_cache:
            return list(self._city_cache[city])
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM pois WHERE city = ? ORDER BY id",
                (city,),
            ).fetchall()
        pois = [self._row_to_poi(row) for row in rows]
        self._city_cache[city] = pois
        return list(pois)

    def load_scope(self, city: str, scope_key: str) -> list[dict[str, Any]] | None:
        cache_key = (city, scope_key)
        if cache_key in self._scope_cache:
            return list(self._scope_cache[cache_key])
        query = "SELECT * FROM pois WHERE city = ?"
        params: list[Any] = [city]
        if scope_key.startswith("zone:"):
            query += " AND zone = ?"
            params.append(scope_key.split(":", 1)[1])
        elif scope_key.startswith("nearby:"):
            zones = [zone for zone in scope_key.split(":", 1)[1].split("|") if zone]
            if not zones:
                return []
            placeholders = ",".join("?" for _ in zones)
            query += f" AND zone IN ({placeholders})"
            params.extend(zones)
        elif scope_key.startswith("district:"):
            query += " AND district = ?"
            params.append(scope_key.split(":", 1)[1])
        elif scope_key == "__all__":
            return self.load_city(city)
        else:
            return None
        query += " ORDER BY id"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        pois = [self._row_to_poi(row) for row in rows]
        self._scope_cache[cache_key] = pois
        return list(pois)

    def load_all_by_city(self) -> dict[str, list[dict[str, Any]]]:
        return {city: self.load_city(city) for city in self.city_counts()}

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _row_to_poi(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            'id': row['id'],
            'name': row['name'],
            'category': row['category'],
            'sub_category': row['sub_category'],
            'city': row['city'],
            'zone': row['zone'],
            'district': row['district'],
            'lat': row['lat'],
            'lng': row['lng'],
            'address': row['address'],
            'rating': row['rating'],
            'price': row['price'],
            'open_time': row['open_time'],
            'close_time': row['close_time'],
            'avg_stay_minutes': row['avg_stay_minutes'],
            'tags': json.loads(row['tags_json'] or '[]'),
            'features': json.loads(row['features_json'] or '{}'),
            'description': row['description'] or '',
            'images': json.loads(row['images_json'] or '[]'),
            'created_at': row['created_at'] or '',
            'updated_at': row['updated_at'] or '',
        }

    def city_counts(self) -> dict[str, int]:
        if self._city_counts is None:
            with self._connect() as conn:
                rows = conn.execute("SELECT city, poi_count FROM cities ORDER BY city").fetchall()
            self._city_counts = {str(row[0]): int(row[1]) for row in rows}
        return dict(self._city_counts)

    def load_city(self, city: str) -> list[dict[str, Any]]:
        if city in self._city_cache:
            return list(self._city_cache[city])
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM pois WHERE city = ? ORDER BY id",
                (city,),
            ).fetchall()
        pois = [self._row_to_poi(row) for row in rows]
        self._city_cache[city] = pois
        return list(pois)

    def load_scope(self, city: str, scope_key: str) -> list[dict[str, Any]] | None:
        cache_key = (city, scope_key)
        if cache_key in self._scope_cache:
            return list(self._scope_cache[cache_key])
        query = "SELECT * FROM pois WHERE city = ?"
        params: list[Any] = [city]
        if scope_key.startswith("zone:"):
            query += " AND zone = ?"
            params.append(scope_key.split(":", 1)[1])
        elif scope_key.startswith("nearby:"):
            zones = [zone for zone in scope_key.split(":", 1)[1].split("|") if zone]
            if not zones:
                return []
            placeholders = ",".join("?" for _ in zones)
            query += f" AND zone IN ({placeholders})"
            params.extend(zones)
        elif scope_key.startswith("district:"):
            query += " AND district = ?"
            params.append(scope_key.split(":", 1)[1])
        elif scope_key == "__all__":
            return self.load_city(city)
        else:
            return None
        query += " ORDER BY id"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        pois = [self._row_to_poi(row) for row in rows]
        self._scope_cache[cache_key] = pois
        return list(pois)

    def _read_json_list(self, path: Path) -> list[dict[str, Any]]:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, list):
            raise ValueError(f"Shard file must contain a JSON list: {path}")
        return [dict(item) for item in payload]


_store_cache: dict[str, PoiStore] = {}


def load_poi_store(source: Path | str) -> PoiStore:
    path = Path(source)
    cache_key = str(path.resolve())
    cached = _store_cache.get(cache_key)
    if cached is not None:
        return cached

    if path.suffix.lower() in {'.db', '.sqlite', '.sqlite3'}:
        store: PoiStore = SqlitePoiStore(path)
    elif path.is_dir() or path.name == MANIFEST_NAME:
        store = PartitionedJsonPoiStore(path)
    else:
        store = MonolithicJsonPoiStore(path)
    _store_cache[cache_key] = store
    return store
