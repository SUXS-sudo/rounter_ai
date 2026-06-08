from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


TimePreference = Literal["morning", "afternoon", "evening", "night", "all_day"]
Pace = Literal["relaxed", "normal", "tight"]
Sentiment = Literal["positive", "neutral", "negative"]


class POIFeatures(BaseModel):
    taste: float = Field(ge=0, le=1, description="Food or drink quality score.")
    photo: float = Field(ge=0, le=1, description="Photo friendliness score.")
    queue_risk: float = Field(ge=0, le=1, description="Probability of queueing or crowding.")
    cost_performance: float = Field(ge=0, le=1, description="Value-for-money score.")
    quiet: float = Field(ge=0, le=1, description="Quietness score.")
    indoor: float = Field(ge=0, le=1, description="Indoor suitability score.")
    family_friendly: float = Field(ge=0, le=1, description="Family friendliness score.")
    night_view: float = Field(ge=0, le=1, description="Night view score.")


class POI(BaseModel):
    id: str
    name: str
    category: str
    sub_category: str
    city: str = "成都"
    zone: str = ""
    lat: float = Field(ge=-90, le=90)
    lng: float = Field(ge=-180, le=180)
    address: str
    rating: float = Field(ge=0, le=5)
    price: int = Field(ge=0, description="Estimated average cost in CNY per person.")
    open_time: str = Field(pattern=r"^\d{2}:\d{2}$")
    close_time: str = Field(pattern=r"^\d{2}:\d{2}$")
    avg_stay_minutes: int = Field(gt=0)
    tags: list[str] = Field(default_factory=list)
    features: POIFeatures


class Review(BaseModel):
    id: str
    poi_id: str
    user_id: str | None = None
    rating: float = Field(ge=0, le=5)
    sentiment: Sentiment = "neutral"
    text: str
    tags: list[str] = Field(default_factory=list)
    created_at: str


class UserProfile(BaseModel):
    id: str
    name: str
    city: str = "成都"
    budget_per_day: int = Field(ge=0)
    preferred_tags: list[str] = Field(default_factory=list)
    disliked_tags: list[str] = Field(default_factory=list)
    favorite_categories: list[str] = Field(default_factory=list)
    dietary_restrictions: list[str] = Field(default_factory=list)
    pace: Pace = "normal"
    time_preference: TimePreference = "all_day"
    feature_weights: POIFeatures


class Location(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lng: float = Field(ge=-180, le=180)
    label: str | None = None


class RouteRequest(BaseModel):
    user_id: str | None = None
    query: str
    start: Location | None = None
    end: Location | None = None
    start_time: str | None = Field(default=None, pattern=r"^\d{2}:\d{2}$")
    duration_minutes: int = Field(default=240, gt=0)
    max_pois: int = Field(default=6, gt=0, le=12)
    budget: int | None = Field(default=None, ge=0)
    required_tags: list[str] = Field(default_factory=list)
    excluded_tags: list[str] = Field(default_factory=list)


class RouteStop(BaseModel):
    poi: POI
    arrive_time: str | None = None
    leave_time: str | None = None
    travel_minutes_from_previous: int = 0
    score: float = Field(default=0, ge=0)
    reason: str | None = None


class RoutePlan(BaseModel):
    request: RouteRequest
    stops: list[RouteStop]
    total_travel_minutes: int = 0
    total_stay_minutes: int = 0
    estimated_cost: int = 0
    explanation: str | None = None
