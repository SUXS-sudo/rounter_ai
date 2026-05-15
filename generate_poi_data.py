#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
POI数据生成器 - 生成10万条模拟POI数据
用于本地出行/游玩路线规划系统
"""

import csv
import json
import random
import sqlite3
import time
import os
from collections import defaultdict
from datetime import datetime

# ============================================================
# 基础数据配置
# ============================================================

# 城市中心坐标 (经度, 维度)
CITIES = {
    "北京": {"center": (116.407, 39.904), "districts": ["朝阳区", "海淀区", "东城区", "西城区", "丰台区", "石景山区", "通州区", "大兴区", "顺义区", "昌平区"]},
    "上海": {"center": (121.473, 31.230), "districts": ["浦东新区", "黄浦区", "静安区", "徐汇区", "长宁区", "虹口区", "杨浦区", "普陀区", "闵行区", "宝山区"]},
    "广州": {"center": (113.264, 23.130), "districts": ["天河区", "越秀区", "海珠区", "荔湾区", "白云区", "番禺区", "黄埔区", "花都区", "南沙区", "增城区"]},
    "深圳": {"center": (114.058, 22.543), "districts": ["南山区", "福田区", "罗湖区", "宝安区", "龙岗区", "龙华区", "光明区", "坪山区", "盐田区", "大鹏新区"]},
    "成都": {"center": (104.066, 30.573), "districts": ["锦江区", "青羊区", "武侯区", "成华区", "金牛区", "高新区", "天府新区", "龙泉驿区", "温江区", "双流区"]},
    "杭州": {"center": (120.155, 30.275), "districts": ["西湖区", "上城区", "拱墅区", "滨江区", "萧山区", "余杭区", "临平区", "富阳区", "临安区", "钱塘区"]},
    "武汉": {"center": (114.305, 30.593), "districts": ["武昌区", "江汉区", "洪山区", "江岸区", "汉阳区", "青山区", "东西湖区", "蔡甸区", "江夏区", "黄陂区"]},
    "西安": {"center": (108.940, 34.265), "districts": ["雁塔区", "碑林区", "莲湖区", "未央区", "灞桥区", "长安区", "高新区", "曲江新区", "新城区", "高陵区"]},
    "重庆": {"center": (106.551, 29.563), "districts": ["渝中区", "江北区", "南岸区", "沙坪坝区", "九龙坡区", "大渡口区", "渝北区", "巴南区", "北碚区", "两江新区"]},
    "南京": {"center": (118.796, 32.060), "districts": ["玄武区", "秦淮区", "建邺区", "鼓楼区", "栖霞区", "雨花台区", "江宁区", "浦口区", "六合区", "溧水区"]},
    "天津": {"center": (117.201, 39.085), "districts": ["和平区", "河西区", "南开区", "河东区", "河北区", "红桥区", "滨海新区", "西青区", "津南区", "东丽区"]},
    "苏州": {"center": (120.585, 31.299), "districts": ["姑苏区", "虎丘区", "吴中区", "相城区", "吴江区", "工业园区", "昆山市", "太仓市", "常熟市", "张家港市"]},
    "长沙": {"center": (112.939, 28.228), "districts": ["芙蓉区", "天心区", "岳麓区", "开福区", "雨花区", "望城区", "长沙县", "浏阳市", "宁乡市"]},
    "青岛": {"center": (120.383, 36.067), "districts": ["市南区", "市北区", "李沧区", "崂山区", "城阳区", "黄岛区", "即墨区", "胶州市", "平度市", "莱西市"]},
    "郑州": {"center": (113.665, 34.758), "districts": ["金水区", "二七区", "中原区", "管城区", "惠济区", "郑东新区", "高新区", "经开区", "航空港区", "上街区"]},
    "厦门": {"center": (118.089, 24.479), "districts": ["思明区", "湖里区", "集美区", "海沧区", "同安区", "翔安区"]},
    "昆明": {"center": (102.833, 25.019), "districts": ["五华区", "盘龙区", "官渡区", "西山区", "呈贡区", "晋宁区", "东川区", "安宁市"]},
    "大连": {"center": (121.615, 38.914), "districts": ["中山区", "西岗区", "沙河口区", "甘井子区", "金州区", "旅顺口区", "普兰店区", "瓦房店市"]},
    "三亚": {"center": (109.512, 18.253), "districts": ["天涯区", "吉阳区", "海棠区", "崖州区"]},
    "丽江": {"center": (100.227, 26.872), "districts": ["古城区", "玉龙县", "永胜县", "华坪县", "宁蒗县"]},
}

# ============================================================
# 热门商圈配置 (zone)
# 每个商圈有坐标中心和辐射半径(度)，POI 生成时 70% 概率落入商圈
# ============================================================

BUSINESS_ZONES: dict[str, list[dict]] = {
    "北京": [
        {"name": "王府井商圈", "center": (116.414, 39.914), "spread": 0.008, "district": "东城区"},
        {"name": "三里屯商圈", "center": (116.455, 39.933), "spread": 0.006, "district": "朝阳区"},
        {"name": "西单商圈", "center": (116.374, 39.910), "spread": 0.007, "district": "西城区"},
        {"name": "国贸商圈", "center": (116.462, 39.909), "spread": 0.008, "district": "朝阳区"},
        {"name": "中关村商圈", "center": (116.316, 39.982), "spread": 0.008, "district": "海淀区"},
        {"name": "五道口商圈", "center": (116.338, 39.992), "spread": 0.005, "district": "海淀区"},
        {"name": "望京商圈", "center": (116.475, 39.998), "spread": 0.007, "district": "朝阳区"},
        {"name": "前门商圈", "center": (116.398, 39.899), "spread": 0.005, "district": "东城区"},
    ],
    "上海": [
        {"name": "南京路商圈", "center": (121.475, 31.235), "spread": 0.006, "district": "黄浦区"},
        {"name": "淮海路商圈", "center": (121.462, 31.220), "spread": 0.006, "district": "黄浦区"},
        {"name": "陆家嘴商圈", "center": (121.499, 31.239), "spread": 0.008, "district": "浦东新区"},
        {"name": "徐家汇商圈", "center": (121.438, 31.188), "spread": 0.007, "district": "徐汇区"},
        {"name": "静安寺商圈", "center": (121.448, 31.223), "spread": 0.006, "district": "静安区"},
        {"name": "五角场商圈", "center": (121.512, 31.303), "spread": 0.007, "district": "杨浦区"},
        {"name": "新天地商圈", "center": (121.475, 31.216), "spread": 0.004, "district": "黄浦区"},
        {"name": "虹桥商圈", "center": (121.408, 31.197), "spread": 0.008, "district": "长宁区"},
    ],
    "广州": [
        {"name": "天河城商圈", "center": (113.328, 23.135), "spread": 0.007, "district": "天河区"},
        {"name": "北京路商圈", "center": (113.269, 23.128), "spread": 0.005, "district": "越秀区"},
        {"name": "珠江新城商圈", "center": (113.320, 23.118), "spread": 0.008, "district": "天河区"},
        {"name": "上下九商圈", "center": (113.244, 23.113), "spread": 0.005, "district": "荔湾区"},
        {"name": "体育中心商圈", "center": (113.335, 23.138), "spread": 0.006, "district": "天河区"},
        {"name": "番禺万博商圈", "center": (113.352, 23.018), "spread": 0.008, "district": "番禺区"},
    ],
    "深圳": [
        {"name": "华强北商圈", "center": (114.088, 22.546), "spread": 0.005, "district": "福田区"},
        {"name": "东门商圈", "center": (114.113, 22.548), "spread": 0.004, "district": "罗湖区"},
        {"name": "南山商圈", "center": (113.930, 22.533), "spread": 0.008, "district": "南山区"},
        {"name": "福田CBD商圈", "center": (114.055, 22.535), "spread": 0.007, "district": "福田区"},
        {"name": "海岸城商圈", "center": (113.942, 22.517), "spread": 0.005, "district": "南山区"},
        {"name": "龙华商圈", "center": (114.023, 22.638), "spread": 0.008, "district": "龙华区"},
    ],
    "成都": [
        {"name": "春熙路商圈", "center": (104.081, 30.657), "spread": 0.005, "district": "锦江区"},
        {"name": "太古里商圈", "center": (104.084, 30.654), "spread": 0.004, "district": "锦江区"},
        {"name": "宽窄巷子商圈", "center": (104.060, 30.670), "spread": 0.004, "district": "青羊区"},
        {"name": "九眼桥商圈", "center": (104.092, 30.641), "spread": 0.005, "district": "锦江区"},
        {"name": "科华路商圈", "center": (104.068, 30.618), "spread": 0.006, "district": "武侯区"},
        {"name": "建设路商圈", "center": (104.105, 30.665), "spread": 0.005, "district": "成华区"},
        {"name": "万象城商圈", "center": (104.098, 30.632), "spread": 0.005, "district": "锦江区"},
        {"name": "天府新区商圈", "center": (104.063, 30.500), "spread": 0.010, "district": "天府新区"},
    ],
    "杭州": [
        {"name": "西湖商圈", "center": (120.148, 30.259), "spread": 0.008, "district": "西湖区"},
        {"name": "武林广场商圈", "center": (120.165, 30.275), "spread": 0.005, "district": "拱墅区"},
        {"name": "湖滨商圈", "center": (120.160, 30.250), "spread": 0.004, "district": "上城区"},
        {"name": "钱江新城商圈", "center": (120.208, 30.250), "spread": 0.008, "district": "上城区"},
        {"name": "西溪商圈", "center": (120.068, 30.270), "spread": 0.008, "district": "西湖区"},
        {"name": "滨江商圈", "center": (120.210, 30.208), "spread": 0.007, "district": "滨江区"},
    ],
    "武汉": [
        {"name": "江汉路商圈", "center": (114.285, 30.583), "spread": 0.005, "district": "江汉区"},
        {"name": "光谷商圈", "center": (114.428, 30.508), "spread": 0.008, "district": "洪山区"},
        {"name": "楚河汉街商圈", "center": (114.345, 30.555), "spread": 0.005, "district": "武昌区"},
        {"name": "武广商圈", "center": (114.268, 30.575), "spread": 0.006, "district": "江汉区"},
        {"name": "街道口商圈", "center": (114.355, 30.530), "spread": 0.005, "district": "洪山区"},
        {"name": "司门口商圈", "center": (114.305, 30.548), "spread": 0.004, "district": "武昌区"},
    ],
    "西安": [
        {"name": "钟楼商圈", "center": (108.940, 34.265), "spread": 0.005, "district": "碑林区"},
        {"name": "小寨商圈", "center": (108.942, 34.228), "spread": 0.005, "district": "雁塔区"},
        {"name": "大雁塔商圈", "center": (108.962, 34.220), "spread": 0.006, "district": "雁塔区"},
        {"name": "回民街商圈", "center": (108.935, 34.265), "spread": 0.003, "district": "莲湖区"},
        {"name": "高新商圈", "center": (108.888, 34.230), "spread": 0.008, "district": "高新区"},
        {"name": "曲江商圈", "center": (108.970, 34.205), "spread": 0.008, "district": "曲江新区"},
    ],
    "重庆": [
        {"name": "解放碑商圈", "center": (106.572, 29.556), "spread": 0.004, "district": "渝中区"},
        {"name": "观音桥商圈", "center": (106.548, 29.575), "spread": 0.006, "district": "江北区"},
        {"name": "沙坪坝商圈", "center": (106.457, 29.540), "spread": 0.006, "district": "沙坪坝区"},
        {"name": "南坪商圈", "center": (106.565, 29.525), "spread": 0.005, "district": "南岸区"},
        {"name": "杨家坪商圈", "center": (106.510, 29.520), "spread": 0.005, "district": "九龙坡区"},
        {"name": "洪崖洞商圈", "center": (106.574, 29.563), "spread": 0.003, "district": "渝中区"},
        {"name": "三峡广场商圈", "center": (106.450, 29.535), "spread": 0.004, "district": "沙坪坝区"},
    ],
    "南京": [
        {"name": "新街口商圈", "center": (118.787, 32.042), "spread": 0.005, "district": "秦淮区"},
        {"name": "夫子庙商圈", "center": (118.790, 32.018), "spread": 0.004, "district": "秦淮区"},
        {"name": "河西商圈", "center": (118.740, 32.028), "spread": 0.008, "district": "建邺区"},
        {"name": "湖南路商圈", "center": (118.772, 32.068), "spread": 0.005, "district": "鼓楼区"},
        {"name": "江宁商圈", "center": (118.840, 31.955), "spread": 0.008, "district": "江宁区"},
    ],
    "天津": [
        {"name": "滨江道商圈", "center": (117.195, 39.120), "spread": 0.005, "district": "和平区"},
        {"name": "小白楼商圈", "center": (117.215, 39.110), "spread": 0.005, "district": "和平区"},
        {"name": "南市商圈", "center": (117.175, 39.130), "spread": 0.004, "district": "南开区"},
        {"name": "意式风情区", "center": (117.200, 39.140), "spread": 0.003, "district": "河北区"},
        {"name": "奥城商圈", "center": (117.150, 39.080), "spread": 0.006, "district": "南开区"},
    ],
    "苏州": [
        {"name": "观前街商圈", "center": (120.620, 31.310), "spread": 0.004, "district": "姑苏区"},
        {"name": "金鸡湖商圈", "center": (120.680, 31.310), "spread": 0.008, "district": "工业园区"},
        {"name": "狮山商圈", "center": (120.560, 31.295), "spread": 0.006, "district": "虎丘区"},
        {"name": "石路商圈", "center": (120.608, 31.315), "spread": 0.004, "district": "姑苏区"},
    ],
    "长沙": [
        {"name": "五一广场商圈", "center": (112.970, 28.200), "spread": 0.005, "district": "芙蓉区"},
        {"name": "太平街商圈", "center": (112.965, 28.195), "spread": 0.003, "district": "天心区"},
        {"name": "IFS商圈", "center": (112.972, 28.198), "spread": 0.004, "district": "芙蓉区"},
        {"name": "梅溪湖商圈", "center": (112.890, 28.180), "spread": 0.008, "district": "岳麓区"},
        {"name": "万家丽商圈", "center": (113.030, 28.210), "spread": 0.006, "district": "芙蓉区"},
    ],
    "青岛": [
        {"name": "台东商圈", "center": (120.350, 36.080), "spread": 0.005, "district": "市北区"},
        {"name": "香港中路商圈", "center": (120.395, 36.060), "spread": 0.006, "district": "市南区"},
        {"name": "中山路商圈", "center": (120.320, 36.065), "spread": 0.004, "district": "市南区"},
        {"name": "崂山商圈", "center": (120.470, 36.105), "spread": 0.008, "district": "崂山区"},
    ],
    "郑州": [
        {"name": "二七广场商圈", "center": (113.655, 34.755), "spread": 0.005, "district": "二七区"},
        {"name": "花园路商圈", "center": (113.680, 34.770), "spread": 0.005, "district": "金水区"},
        {"name": "郑东新区CBD", "center": (113.720, 34.760), "spread": 0.008, "district": "郑东新区"},
        {"name": "紫荆山商圈", "center": (113.665, 34.758), "spread": 0.004, "district": "金水区"},
    ],
    "厦门": [
        {"name": "中山路商圈", "center": (118.075, 24.450), "spread": 0.004, "district": "思明区"},
        {"name": "SM商圈", "center": (118.110, 24.500), "spread": 0.005, "district": "湖里区"},
        {"name": "曾厝垵商圈", "center": (118.098, 24.440), "spread": 0.003, "district": "思明区"},
        {"name": "鼓浪屿", "center": (118.065, 24.445), "spread": 0.003, "district": "思明区"},
    ],
    "昆明": [
        {"name": "南屏街商圈", "center": (102.715, 25.035), "spread": 0.004, "district": "五华区"},
        {"name": "翠湖商圈", "center": (102.700, 25.050), "spread": 0.005, "district": "五华区"},
        {"name": "金马碧鸡坊", "center": (102.710, 25.030), "spread": 0.003, "district": "西山区"},
        {"name": "呈贡商圈", "center": (102.830, 24.880), "spread": 0.008, "district": "呈贡区"},
    ],
    "大连": [
        {"name": "青泥洼桥商圈", "center": (121.630, 38.915), "spread": 0.004, "district": "中山区"},
        {"name": "西安路商圈", "center": (121.590, 38.920), "spread": 0.005, "district": "沙河口区"},
        {"name": "星海广场商圈", "center": (121.570, 38.880), "spread": 0.006, "district": "沙河口区"},
        {"name": "东港商圈", "center": (121.650, 38.920), "spread": 0.005, "district": "中山区"},
    ],
    "三亚": [
        {"name": "三亚湾商圈", "center": (109.490, 18.250), "spread": 0.008, "district": "天涯区"},
        {"name": "亚龙湾商圈", "center": (109.620, 18.180), "spread": 0.006, "district": "吉阳区"},
        {"name": "海棠湾商圈", "center": (109.720, 18.310), "spread": 0.008, "district": "海棠区"},
        {"name": "大东海商圈", "center": (109.515, 18.220), "spread": 0.004, "district": "吉阳区"},
    ],
    "丽江": [
        {"name": "大研古城", "center": (100.227, 26.872), "spread": 0.005, "district": "古城区"},
        {"name": "束河古镇", "center": (100.210, 26.890), "spread": 0.004, "district": "古城区"},
        {"name": "忠义市场商圈", "center": (100.230, 26.865), "spread": 0.003, "district": "古城区"},
    ],
}

# ============================================================
# POI类别配置
# ============================================================

CATEGORIES = {
    "餐饮": {
        "subcategories": ["中餐", "西餐", "日料", "韩餐", "火锅", "烧烤", "小吃快餐", "甜品饮品", "奶茶店", "自助餐", "海鲜", "川菜", "粤菜", "湘菜", "江浙菜", "东北菜", "云南菜", "面馆", "咖啡厅", "茶馆", "酒吧"],
        "avg_cost_range": (15, 800),
        "queue_range": (0, 120),
        "rating_bias": 3.8,
        "weight": 35,  # 占比权重
    },
    "景点": {
        "subcategories": ["自然风光", "历史古迹", "主题公园", "博物馆", "美术馆", "动物园", "植物园", "海洋馆", "古镇古村", "城市地标", "公园", "寺庙", "纪念馆", "展览馆", "观光塔"],
        "avg_cost_range": (0, 500),
        "queue_range": (0, 180),
        "rating_bias": 4.0,
        "weight": 20,
    },
    "娱乐": {
        "subcategories": ["电影院", "KTV", "桌游密室", "剧本杀", "电玩城", "游乐园", "水上乐园", "滑雪场", "高尔夫", "保龄球", "射箭馆", "蹦床公园", "VR体验", "真人CS", "温泉"],
        "avg_cost_range": (30, 600),
        "queue_range": (0, 90),
        "rating_bias": 3.9,
        "weight": 15,
    },
    "文化": {
        "subcategories": ["图书馆", "剧院", "音乐厅", "画廊", "文创园区", "书店", "非遗体验", "手工艺坊", "摄影基地", "文化广场"],
        "avg_cost_range": (0, 300),
        "queue_range": (0, 60),
        "rating_bias": 4.2,
        "weight": 10,
    },
    "购物": {
        "subcategories": ["大型商场", "购物中心", "步行街", "奥特莱斯", "超市", "便利店", "特产店", "花鸟市场", "古玩市场", "夜市"],
        "avg_cost_range": (0, 2000),
        "queue_range": (0, 45),
        "rating_bias": 3.7,
        "weight": 10,
    },
    "运动健身": {
        "subcategories": ["健身房", "游泳馆", "篮球场", "足球场", "网球场", "羽毛球馆", "瑜伽馆", "攀岩馆", "搏击馆", "滑板公园", "骑行道", "跑步公园"],
        "avg_cost_range": (20, 300),
        "queue_range": (0, 30),
        "rating_bias": 4.0,
        "weight": 5,
    },
    "亲子": {
        "subcategories": ["儿童乐园", "亲子餐厅", "科普馆", "少年宫", "儿童剧场", "采摘园", "牧场体验", "手工DIY", "水上乐园", "拓展训练"],
        "avg_cost_range": (30, 400),
        "queue_range": (0, 90),
        "rating_bias": 4.1,
        "weight": 5,
    },
}

CATEGORY_EN_MAP = {
    "餐饮": "food",
    "景点": "attraction",
    "娱乐": "entertainment",
    "文化": "culture",
    "购物": "shopping",
    "运动健身": "fitness",
    "亲子": "family",
}

SUBCATEGORY_EN_MAP = {
    "餐饮": {
        "中餐": "chinese", "西餐": "western", "日料": "japanese", "韩餐": "korean",
        "火锅": "hotpot", "烧烤": "bbq", "小吃快餐": "snack", "甜品饮品": "dessert", "奶茶店": "milk_tea",
        "自助餐": "buffet", "海鲜": "seafood", "川菜": "sichuan", "粤菜": "cantonese",
        "湘菜": "hunan", "江浙菜": "jiangzhe", "东北菜": "northeast", "云南菜": "yunnan",
        "面馆": "noodles", "咖啡厅": "cafe", "茶馆": "teahouse", "酒吧": "bar",
    },
    "景点": {
        "自然风光": "natural", "历史古迹": "historic", "主题公园": "theme_park",
        "博物馆": "museum", "美术馆": "gallery", "动物园": "zoo", "植物园": "botanical",
        "海洋馆": "aquarium", "古镇古村": "ancient_town", "城市地标": "landmark",
        "公园": "park", "寺庙": "temple", "纪念馆": "memorial", "展览馆": "exhibition",
        "观光塔": "tower",
    },
    "娱乐": {
        "电影院": "cinema", "KTV": "ktv", "桌游密室": "board_game", "剧本杀": "script_kill",
        "电玩城": "arcade", "游乐园": "amusement_park", "水上乐园": "water_park",
        "滑雪场": "ski_resort", "高尔夫": "golf", "保龄球": "bowling", "射箭馆": "archery",
        "蹦床公园": "trampoline", "VR体验": "vr_experience", "真人CS": "paintball",
        "温泉": "hot_spring",
    },
    "文化": {
        "图书馆": "library", "剧院": "theater", "音乐厅": "concert_hall",
        "画廊": "art_gallery", "文创园区": "creative_park", "书店": "bookstore",
        "非遗体验": "heritage", "手工艺坊": "craft_workshop", "摄影基地": "photo_studio",
        "文化广场": "culture_square",
    },
    "购物": {
        "大型商场": "mall", "购物中心": "shopping_center", "步行街": "pedestrian_street",
        "奥特莱斯": "outlet", "超市": "supermarket", "便利店": "convenience",
        "特产店": "souvenir", "花鸟市场": "flower_market", "古玩市场": "antique_market",
        "夜市": "night_market",
    },
    "运动健身": {
        "健身房": "gym", "游泳馆": "swimming", "篮球场": "basketball", "足球场": "football",
        "网球场": "tennis", "羽毛球馆": "badminton", "瑜伽馆": "yoga", "攀岩馆": "climbing",
        "搏击馆": "martial_arts", "滑板公园": "skate_park", "骑行道": "cycling",
        "跑步公园": "running",
    },
    "亲子": {
        "儿童乐园": "kids_play", "亲子餐厅": "family_restaurant", "科普馆": "science_center",
        "少年宫": "children_palace", "儿童剧场": "kids_theater", "采摘园": "picking_garden",
        "牧场体验": "ranch", "手工DIY": "diy_workshop", "水上乐园": "water_park",
        "拓展训练": "outdoor_training",
    },
}

FEATURES_BIAS = {
    "餐饮": {"taste": (0.55, 0.92), "photo": (0.3, 0.75), "queue_risk": (0.4, 0.85),
             "cost_performance": (0.4, 0.85), "quiet": (0.1, 0.5), "indoor": (0.6, 1.0),
             "family_friendly": (0.35, 0.75), "night_view": (0.15, 0.55)},
    "景点": {"taste": (0.05, 0.3), "photo": (0.6, 0.98), "queue_risk": (0.25, 0.8),
             "cost_performance": (0.5, 0.95), "quiet": (0.4, 0.85), "indoor": (0.05, 0.45),
             "family_friendly": (0.55, 0.95), "night_view": (0.3, 0.75)},
    "娱乐": {"taste": (0.05, 0.25), "photo": (0.35, 0.75), "queue_risk": (0.3, 0.75),
             "cost_performance": (0.35, 0.75), "quiet": (0.05, 0.35), "indoor": (0.55, 1.0),
             "family_friendly": (0.3, 0.7), "night_view": (0.1, 0.45)},
    "文化": {"taste": (0.05, 0.2), "photo": (0.55, 0.9), "queue_risk": (0.05, 0.45),
             "cost_performance": (0.6, 0.95), "quiet": (0.65, 1.0), "indoor": (0.6, 1.0),
             "family_friendly": (0.35, 0.7), "night_view": (0.05, 0.35)},
    "购物": {"taste": (0.05, 0.3), "photo": (0.3, 0.7), "queue_risk": (0.2, 0.65),
             "cost_performance": (0.35, 0.75), "quiet": (0.05, 0.35), "indoor": (0.65, 1.0),
             "family_friendly": (0.5, 0.85), "night_view": (0.25, 0.65)},
    "运动健身": {"taste": (0.05, 0.2), "photo": (0.1, 0.4), "queue_risk": (0.05, 0.35),
                 "cost_performance": (0.4, 0.8), "quiet": (0.3, 0.7), "indoor": (0.5, 1.0),
                 "family_friendly": (0.1, 0.45), "night_view": (0.05, 0.25)},
    "亲子": {"taste": (0.2, 0.55), "photo": (0.4, 0.8), "queue_risk": (0.25, 0.7),
             "cost_performance": (0.4, 0.8), "quiet": (0.3, 0.65), "indoor": (0.35, 0.85),
             "family_friendly": (0.8, 1.0), "night_view": (0.05, 0.3)},
}

# ============================================================
# POI名称生成素材
# ============================================================

# 餐饮名称组件
RESTAURANT_PREFIXES = ["老", "小", "大", "金", "银", "红", "绿", "新", "古", "鲜", "香", "辣", "蜀", "粤", "湘", "江南", "塞北", "东海", "西域", "南国", "北国", "天府", "巴蜀", "岭南", "齐鲁", "关中", "江南", "徽州", "闽南", "潮汕"]
RESTAURANT_MIDS = ["张", "李", "王", "刘", "陈", "杨", "赵", "黄", "周", "吴", "徐", "孙", "马", "朱", "胡", "郭", "林", "何", "高", "罗", "郑", "梁", "谢", "宋", "唐", "韩", "冯", "董", "程", "蔡"]
RESTAURANT_SUFFIXES_RESTAURANT = ["餐厅", "饭店", "酒楼", "食府", "小馆", "私房菜", "家常菜", "美食城", "菜馆", "饭庄"]
RESTAURANT_SUFFIXES_HOTPOT = ["火锅", "火锅城", "火锅店", "涮肉坊", "铜锅涮"]
RESTAURANT_SUFFIXES_BBQ = ["烧烤", "烤肉", "烤鱼", "铁板烧", "炭火烤肉"]
RESTAURANT_SUFFIXES_CAFE = ["咖啡", "咖啡馆", "咖啡厅", "茶室", "茶馆", "茶舍", "奶茶店", "甜品店", "烘焙坊"]
MILK_TEA_BRANDS = ["蜜雪冰城", "古茗", "奈雪的茶", "茶百道", "霸王茶姬", "书亦烧仙草", "沪上阿姨", "一点点", "CoCo都可", "益禾堂"]
LOCAL_TEA_PREFIXES = ["巷口", "街角", "本地", "阿姨", "现萃", "手作", "山野", "果语", "茶里", "茶叙", "青柠", "橘子海"]
LOCAL_TEA_SUFFIXES = ["奶茶", "奶茶铺", "茶饮店", "鲜果茶", "手作茶饮", "柠檬茶", "芋泥茶铺", "杨枝甘露站", "波波冰铺"]
RESTAURANT_SUFFIXES_SNACK = ["小吃", "面馆", "粉店", "饺子馆", "包子铺", "煎饼店", "麻辣烫", "米线店", "馄饨店", "粥铺"]

# 景点名称组件
ATTRACTION_TYPES = ["公园", "花园", "景区", "风景区", "旅游区", "度假区", "生态园", "湿地公园", "森林公园", "地质公园", "遗址", "古迹", "故居", "旧址", "纪念馆", "博物院", "展览馆", "美术馆", "科技馆", "天文馆"]
ATTRACTION_PREFIXES = ["国家", "省级", "市级", "东方", "西方", "南方", "北方", "中原", "江南", "塞北", "东海", "西域", "南国", "北国", "天府", "巴蜀", "岭南", "齐鲁", "关中", "徽州", "闽南", "潮汕", "华夏", "神州", "九州", "中华", "龙腾", "凤舞", "山水", "云端", "星空", "月光", "阳光", "清风", "明月", "碧波", "翠竹", "松涛", "梅兰", "竹菊", "桃源", "仙境", "世外", "桃源"]

# 娱乐名称组件
ENTERTAINMENT_PREFIXES = ["欢乐", "开心", "快乐", "疯狂", "奇妙", "梦幻", "星际", "未来", "超级", "酷玩", "乐翻天", "嗨翻天", "玩转", "趣玩", "妙趣", "奇趣", "乐动", "动感", "激情", "飞扬"]

# 通用形容词
ADJECTIVES = ["优雅", "时尚", "经典", "精致", "高端", "大气", "温馨", "浪漫", "复古", "现代", "简约", "奢华", "清幽", "热闹", "繁华", "宁静", "古朴", "典雅", "别致", "独特"]

# 街道名称
STREET_NAMES = ["中山路", "人民路", "解放路", "建设路", "文化路", "和平路", "胜利路", "光明路", "新华路", "长安街", "南京路", "淮海路", "王府井", "春熙路", "上下九", "解放碑", "江汉路", "户部巷", "回民街", "夫子庙", "城隍庙", "步行街", "美食街", "商业街", "小吃街", "文化街", "古镇街", "老街", "新街", "大街", "大道", "广场路", "公园路", "学校路", "医院路", "科技路", "创新路", "发展路", "前进路", "幸福路", "和谐路", "团结路", "友谊路", "爱国路", "敬业路", "诚信路", "友善路"]

# ============================================================
# 标签配置
# ============================================================

TAGS_BY_CATEGORY = {
    "餐饮": ["环境好", "味道好", "分量足", "性价比高", "服务好", "有包间", "可停车", "有WiFi", "24小时营业", "老字号", "网红店", "排队少", "适合聚餐", "适合约会", "适合商务", "适合家庭", "有儿童椅", "可外卖", "有露台", "景观位"],
    "景点": ["免费", "拍照圣地", "遛娃好去处", "适合徒步", "历史底蕴", "自然风光", "网红打卡", "人少景美", "交通方便", "有讲解", "有导游", "适合老人", "适合情侣", "适合团建", "四季皆宜", "春天赏花", "秋天赏叶", "冬天赏雪", "夏天避暑"],
    "娱乐": ["刺激", "解压", "适合团建", "适合约会", "适合朋友聚会", "有教练", "设备新", "场地大", "有空调", "有储物柜", "可团购", "有会员卡", "周末活动", "节假日优惠"],
    "文化": ["文艺", "小众", "有展览", "有活动", "适合学习", "适合拍照", "安静", "免费", "有讲座", "有工作坊", "有纪念品", "有咖啡"],
    "购物": ["品牌齐全", "打折", "免税", "有停车场", "交通方便", "有餐饮", "有影院", "有儿童区", "有休息区", "有WiFi", "有充电桩"],
    "运动健身": ["器械齐全", "有教练", "有淋浴", "有储物柜", "有空调", "场地好", "灯光好", "可预约", "有会员卡", "有团购"],
    "亲子": ["安全", "有监护人陪同区", "适合0-3岁", "适合3-6岁", "适合6-12岁", "有休息区", "有餐饮", "有卫生间", "有停车位", "有活动"],
}

# 营业时间模板
OPENING_HOURS_TEMPLATES = [
    "09:00-22:00", "08:00-21:00", "10:00-22:00", "09:30-21:30",
    "08:30-20:30", "10:00-21:00", "09:00-21:00", "10:00-23:00",
    "11:00-23:00", "10:00-18:00", "08:00-17:30", "09:00-17:00",
    "06:00-22:00", "07:00-21:00", "24小时营业", "14:00-02:00",
    "16:00-00:00", "09:00-18:00", "10:00-20:00", "08:00-22:00",
]

# 描述模板
DESCRIPTION_TEMPLATES = {
    "餐饮": [
        "{name}位于{city}{district}，是一家主营{subcategory}的餐厅。餐厅环境{adj}，菜品口味正宗，食材新鲜，深受本地食客喜爱。招牌菜色香味俱全，{tag}，是{scene}的绝佳选择。",
        "{name}坐落于{city}{district}{street}，以{subcategory}闻名。店内装修{adj}，服务周到热情，菜品分量十足且价格实惠。{tag}，多年来积累了大量忠实顾客。",
        "位于{city}{district}的{name}，专注于{subcategory}美食。餐厅氛围{adj}，每道菜品都精心烹制，色香味俱佳。{tag}，是当地人气餐厅之一。",
    ],
    "景点": [
        "{name}位于{city}{district}，是{city}知名的{subcategory}景点。景区环境优美，空气清新，{tag}。一年四季皆有不同的风景，是休闲度假的好去处。",
        "坐落于{city}{district}的{name}，以其独特的{subcategory}魅力吸引着众多游客。这里{adj}静谧，{tag}，是感受{city}文化底蕴的绝佳之地。",
        "{name}是{city}{district}的标志性{subcategory}景点。这里{adj}大气，景色宜人，{tag}。无论是本地居民还是外地游客，都会被这里的美景所吸引。",
    ],
    "娱乐": [
        "{name}位于{city}{district}，是当地热门的{subcategory}娱乐场所。设施{adj}，项目丰富多样，{tag}。是朋友聚会、情侣约会的不二之选。",
        "坐落于{city}{district}{street}的{name}，提供专业的{subcategory}体验。场馆环境{adj}，设备先进，{tag}，让你尽情释放压力。",
        "{name}是{city}{district}新兴的{subcategory}娱乐地标。装修风格{adj}，氛围感十足，{tag}，深受年轻人喜爱。",
    ],
    "文化": [
        "{name}位于{city}{district}，是{city}重要的{subcategory}文化场所。空间{adj}典雅，藏品丰富，{tag}。定期举办各类文化活动，是感受艺术熏陶的好去处。",
        "坐落于{city}{district}的{name}，以{subcategory}为主题。环境{adj}，文化氛围浓厚，{tag}，是文艺青年的打卡圣地。",
    ],
    "购物": [
        "{name}位于{city}{district}{street}，是{city}知名的{subcategory}购物场所。品牌齐全，环境{adj}，{tag}。是购物休闲的理想之地。",
        "坐落于{city}{district}的{name}，是集购物、餐饮、娱乐于一体的{subcategory}。设施{adj}，服务周到，{tag}，满足一站式消费需求。",
    ],
    "运动健身": [
        "{name}位于{city}{district}，是专业的{subcategory}运动场所。场馆{adj}，器材齐全，{tag}。无论是初学者还是运动达人，都能在这里找到适合自己的项目。",
        "坐落于{city}{district}的{name}，提供高品质{subcategory}服务。环境{adj}，教练专业，{tag}，是健身爱好者的首选之地。",
    ],
    "亲子": [
        "{name}位于{city}{district}，是{city}受欢迎的{subcategory}亲子场所。环境安全{adj}，项目寓教于乐，{tag}。是家长带娃出行的放心之选。",
        "坐落于{city}{district}的{name}，专注于{subcategory}亲子体验。设施{adj}，服务贴心，{tag}，让家长和孩子都能享受美好时光。",
    ],
}

# 使用场景
SCENES = {
    "餐饮": ["朋友聚餐", "家庭聚会", "商务宴请", "情侣约会", "同事小聚", "生日庆祝", "节日聚餐", "日常用餐"],
    "景点": ["周末出游", "节假日旅行", "摄影采风", "亲子活动", "朋友聚会", "情侣出游", "团队活动"],
    "娱乐": ["朋友聚会", "情侣约会", "公司团建", "生日派对", "周末休闲", "解压放松"],
    "文化": ["文艺打卡", "学习充电", "周末休闲", "约会", "亲子活动", "朋友聚会"],
    "购物": ["日常购物", "逛街休闲", "买礼物", "家庭采购", "节日购物"],
    "运动健身": ["日常锻炼", "减脂塑形", "增肌训练", "放松身心", "学习技能"],
    "亲子": ["亲子活动", "周末遛娃", "生日派对", "假期活动", "学习体验"],
}

# ============================================================
# 生成函数
# ============================================================


def weighted_choice(categories_dict):
    """按权重选择类别"""
    total = sum(v["weight"] for v in categories_dict.values())
    r = random.uniform(0, total)
    cumulative = 0
    for cat, info in categories_dict.items():
        cumulative += info["weight"]
        if r <= cumulative:
            return cat
    return list(categories_dict.keys())[0]


def generate_coordinate(center, spread=0.03):
    """生成以城市中心为基准的随机坐标"""
    lon = center[0] + random.uniform(-spread, spread)
    lat = center[1] + random.uniform(-spread, spread)
    return round(lon, 6), round(lat, 6)


def pick_zone(city_name, category=None, subcategory=None):
    """为 POI 选择商圈。餐饮尤其是奶茶/咖啡/茶饮更容易落入热门商圈。"""
    zones = BUSINESS_ZONES.get(city_name, [])
    if not zones:
        return None

    zone_probability = 0.70
    if category == "餐饮":
        zone_probability = 0.82
        if subcategory in {"奶茶店", "甜品饮品", "咖啡厅", "茶馆"}:
            zone_probability = 0.92

    if random.random() < zone_probability:
        return random.choice(zones)
    return None


def generate_phone():
    """生成随机手机号"""
    prefixes = ["130", "131", "132", "133", "134", "135", "136", "137", "138", "139",
                 "150", "151", "152", "153", "155", "156", "157", "158", "159",
                 "170", "171", "172", "173", "175", "176", "177", "178",
                 "180", "181", "182", "183", "184", "185", "186", "187", "188", "189"]
    return random.choice(prefixes) + "".join([str(random.randint(0, 9)) for _ in range(8)])


def generate_poi_name(category, subcategory):
    """根据类别生成POI名称"""
    if category == "餐饮":
        if subcategory == "奶茶店":
            if random.random() < 0.62:
                return random.choice(MILK_TEA_BRANDS)
            return f"{random.choice(LOCAL_TEA_PREFIXES)}{random.choice(LOCAL_TEA_SUFFIXES)}"
        if "火锅" in subcategory:
            prefix = random.choice(RESTAURANT_PREFIXES + RESTAURANT_MIDS)
            suffix = random.choice(RESTAURANT_SUFFIXES_HOTPOT)
            return f"{prefix}{suffix}"
        elif "烧烤" in subcategory:
            prefix = random.choice(RESTAURANT_PREFIXES + RESTAURANT_MIDS)
            suffix = random.choice(RESTAURANT_SUFFIXES_BBQ)
            return f"{prefix}{suffix}"
        elif any(x in subcategory for x in ["咖啡", "甜品", "饮品", "茶"]):
            prefix = random.choice(RESTAURANT_PREFIXES + RESTAURANT_MIDS + ADJECTIVES)
            suffix = random.choice(RESTAURANT_SUFFIXES_CAFE)
            return f"{prefix}{suffix}"
        elif any(x in subcategory for x in ["小吃", "面", "粉", "快餐"]):
            prefix = random.choice(RESTAURANT_PREFIXES + RESTAURANT_MIDS)
            suffix = random.choice(RESTAURANT_SUFFIXES_SNACK)
            return f"{prefix}{suffix}"
        else:
            prefix = random.choice(RESTAURANT_PREFIXES + RESTAURANT_MIDS)
            suffix = random.choice(RESTAURANT_SUFFIXES_RESTAURANT)
            return f"{prefix}{suffix}"
    elif category == "景点":
        prefix = random.choice(ATTRACTION_PREFIXES)
        suffix = random.choice(ATTRACTION_TYPES)
        return f"{prefix}{suffix}"
    elif category == "娱乐":
        prefix = random.choice(ENTERTAINMENT_PREFIXES)
        return f"{prefix}{subcategory}"
    elif category == "文化":
        prefix = random.choice(ADJECTIVES + ["新", "旧", "古", "今", "雅", "韵", "墨", "书", "艺", "文"])
        return f"{prefix}{subcategory}"
    elif category == "购物":
        prefix = random.choice(["万达", "银泰", "大悦城", "华润", "龙湖", "中粮", "凯德", "恒隆", "太古", "新世界", "百联", "天虹", "茂业", "金鹰", "来福士", "印象城", "吾悦", "宝龙", "融创", "新城"])
        return f"{prefix}{random.choice(['广场', '中心', 'mall', '荟', '天地'])}"
    elif category == "运动健身":
        prefix = random.choice(["活力", "动感", "极速", "飞扬", "超越", "巅峰", "精英", "冠军", "力量", "速度", "耐力", "爆发", "热血", "激情", "阳光", "青春"])
        return f"{prefix}{subcategory}"
    elif category == "亲子":
        prefix = random.choice(["小天才", "快乐宝贝", "童趣", "乐高", "梦幻", "奇妙", "开心果", "小星星", "月亮船", "彩虹桥", "太阳花", "向日葵", "小蜜蜂", "蝴蝶谷", "米奇", "迪士尼"])
        return f"{prefix}{subcategory}"
    return f"{random.choice(ADJECTIVES)}{subcategory}"


def generate_description(name, city, district, street, category, subcategory, tags):
    """生成POI描述"""
    templates = DESCRIPTION_TEMPLATES.get(category, DESCRIPTION_TEMPLATES["景点"])
    template = random.choice(templates)
    adj = random.choice(ADJECTIVES)
    tag_str = "、".join(random.sample(tags, min(3, len(tags))))
    scene = random.choice(SCENES.get(category, ["休闲"]))
    return template.format(
        name=name, city=city, district=district, street=street,
        subcategory=subcategory, adj=adj, tag=tag_str, scene=scene
    )


def generate_features(category, subcategory=None):
    """根据类别生成 features 特征向量"""
    bias = FEATURES_BIAS.get(category, FEATURES_BIAS["景点"])
    features = {}
    for key, (lo, hi) in bias.items():
        val = random.uniform(lo, hi)
        features[key] = round(min(1.0, max(0.0, val)), 2)

    if category == "餐饮" and subcategory == "奶茶店":
        features["cost_performance"] = round(max(features.get("cost_performance", 0.6), random.uniform(0.72, 0.96)), 2)
        features["queue_risk"] = round(min(features.get("queue_risk", 0.55), random.uniform(0.08, 0.45)), 2)
        features["photo"] = round(max(features.get("photo", 0.45), random.uniform(0.45, 0.85)), 2)
        features["indoor"] = round(max(features.get("indoor", 0.75), random.uniform(0.82, 1.0)), 2)
    if category == "餐饮" and subcategory == "火锅":
        features["taste"] = round(max(features.get("taste", 0.65), random.uniform(0.78, 0.98)), 2)
        features["indoor"] = round(max(features.get("indoor", 0.7), random.uniform(0.8, 1.0)), 2)
        features["cost_performance"] = round(max(features.get("cost_performance", 0.55), random.uniform(0.58, 0.9)), 2)
    return features


def generate_poi(poi_id, cities, categories):
    """生成单条POI数据"""
    city_name = random.choice(list(cities.keys()))
    city_info = cities[city_name]

    category = weighted_choice(categories)
    cat_info = categories[category]
    subcategories = list(cat_info["subcategories"])
    if category == "餐饮":
        weighted_subcategories = []
        for subcategory_name in subcategories:
            weight = 1
            if subcategory_name in {"甜品饮品", "咖啡厅", "茶馆"}:
                weight = 2
            if subcategory_name == "奶茶店":
                weight = 3
            weighted_subcategories.extend([subcategory_name] * weight)
        subcategory = random.choice(weighted_subcategories)
    else:
        subcategory = random.choice(subcategories)

    zone_info = pick_zone(city_name, category=category, subcategory=subcategory)
    if zone_info:
        zone_name = zone_info["name"]
        district = zone_info["district"]
        lon, lat = generate_coordinate(zone_info["center"], zone_info["spread"])
        if category == "餐饮" and random.random() < 0.55:
            subcategory = random.choice(["奶茶店", "甜品饮品", "咖啡厅", "茶馆"])
    else:
        zone_name = ""
        district = random.choice(city_info["districts"])
        lon, lat = generate_coordinate(city_info["center"])

    name = generate_poi_name(category, subcategory)

    street = random.choice(STREET_NAMES)
    street_num = random.randint(1, 999)
    address = f"{city_name}{district}{street}{street_num}号"

    base_rating = cat_info["rating_bias"]
    rating = round(min(5.0, max(1.0, random.gauss(base_rating, 0.5))), 1)

    cost_min, cost_max = cat_info["avg_cost_range"]
    price = round(random.uniform(cost_min, cost_max), 0)

    q_min, q_max = cat_info["queue_range"]

    opening_hours = random.choice(OPENING_HOURS_TEMPLATES)
    if "-" in opening_hours:
        parts = opening_hours.split("-")
        open_time = parts[0]
        close_time = parts[1]
    else:
        open_time = "00:00"
        close_time = "23:59"

    avg_stay_minutes = random.randint(20, 180)

    available_tags = TAGS_BY_CATEGORY.get(category, ["热门"])
    tags = random.sample(available_tags, min(random.randint(2, 5), len(available_tags)))
    if category == "餐饮":
        food_tags = ["适合休息", "可久坐", "室内", "逛街补给", "排队少"]
        tags = list(dict.fromkeys(tags + random.sample(food_tags, k=min(2, len(food_tags)))))
    if category == "餐饮" and subcategory == "火锅":
        hotpot_tags = ["火锅", "热乎", "聚餐", "锅底", "川味", "麻辣"]
        tags = list(dict.fromkeys(tags + random.sample(hotpot_tags, k=min(3, len(hotpot_tags)))))
    if category == "餐饮" and subcategory == "奶茶店":
        tea_tags = ["奶茶", "茶饮", "果茶", "下午茶", "平价", "适合外带", "逛街补给", "排队少"]
        tags = list(dict.fromkeys(tags + random.sample(tea_tags, k=min(3, len(tea_tags)))))

    description = generate_description(name, city_name, district, street, category, subcategory, tags)

    image_count = random.randint(1, 9)
    images = [f"https://pics.example.com/poi/{poi_id}_{i}.jpg" for i in range(image_count)]

    features = generate_features(category, subcategory=subcategory)

    category_en = CATEGORY_EN_MAP.get(category, "attraction")
    sub_category_en = SUBCATEGORY_EN_MAP.get(category, {}).get(subcategory, subcategory)

    return {
        "id": f"gen_{poi_id:07d}",
        "name": name,
        "category": category_en,
        "sub_category": sub_category_en,
        "city": city_name,
        "zone": zone_name,
        "lat": lat,
        "lng": lon,
        "address": address,
        "rating": rating,
        "price": int(price),
        "open_time": open_time,
        "close_time": close_time,
        "avg_stay_minutes": avg_stay_minutes,
        "tags": tags,
        "features": features,
        "district": district,
        "description": description,
        "images": images,
        "created_at": f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
        "updated_at": f"2025-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
    }


def write_csv(data, filepath):
    """写入CSV文件"""
    if not data:
        return
    fieldnames = data[0].keys()
    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            # 将列表和字典转为JSON字符串
            row_copy = {}
            for k, v in row.items():
                if isinstance(v, (list, dict)):
                    row_copy[k] = json.dumps(v, ensure_ascii=False)
                else:
                    row_copy[k] = v
            writer.writerow(row_copy)


def write_json(data, filepath):
    """写入JSON文件"""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _safe_path_name(value: str) -> str:
    return (
        str(value)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("*", "_")
        .replace("?", "_")
        .replace('"', "_")
        .replace("<", "_")
        .replace(">", "_")
        .replace("|", "_")
    )



def _zone_lookup() -> dict[str, dict[str, dict]]:
    return {
        city: {zone['name']: zone for zone in zones}
        for city, zones in BUSINESS_ZONES.items()
    }



def write_sqlite(data: list[dict], output_dir: str) -> str:
    db_path = os.path.join(output_dir, 'poi_data_500k.db')
    if os.path.exists(db_path):
        os.remove(db_path)

    zone_lookup = _zone_lookup()
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE cities (
                city TEXT PRIMARY KEY,
                poi_count INTEGER NOT NULL
            );
            CREATE TABLE zones (
                city TEXT NOT NULL,
                zone TEXT NOT NULL,
                district TEXT,
                center_lng REAL,
                center_lat REAL,
                poi_count INTEGER NOT NULL,
                shard_aliases_json TEXT NOT NULL,
                PRIMARY KEY (city, zone)
            );
            CREATE TABLE districts (
                city TEXT NOT NULL,
                district TEXT NOT NULL,
                poi_count INTEGER NOT NULL,
                PRIMARY KEY (city, district)
            );
            CREATE TABLE pois (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                sub_category TEXT NOT NULL,
                city TEXT NOT NULL,
                zone TEXT,
                district TEXT,
                lat REAL NOT NULL,
                lng REAL NOT NULL,
                address TEXT NOT NULL,
                rating REAL NOT NULL,
                price INTEGER NOT NULL,
                open_time TEXT NOT NULL,
                close_time TEXT NOT NULL,
                avg_stay_minutes INTEGER NOT NULL,
                tags_json TEXT NOT NULL,
                features_json TEXT NOT NULL,
                description TEXT,
                images_json TEXT NOT NULL,
                created_at TEXT,
                updated_at TEXT
            );
            CREATE INDEX idx_pois_city ON pois(city);
            CREATE INDEX idx_pois_city_zone ON pois(city, zone);
            CREATE INDEX idx_pois_city_district ON pois(city, district);
            """
        )

        city_counts = defaultdict(int)
        district_counts = defaultdict(int)
        zone_counts = defaultdict(int)
        zone_aliases = defaultdict(set)
        poi_rows = []
        for poi in data:
            city = str(poi['city'])
            zone = str(poi.get('zone') or '')
            district = str(poi.get('district') or '')
            city_counts[city] += 1
            if district:
                district_counts[(city, district)] += 1
            if zone:
                zone_counts[(city, zone)] += 1
                zone_aliases[(city, zone)].add(zone)
                if zone.endswith('商圈'):
                    zone_aliases[(city, zone)].add(zone[:-2])
            poi_rows.append((
                poi['id'], poi['name'], poi['category'], poi['sub_category'], city, zone, district,
                poi['lat'], poi['lng'], poi['address'], poi['rating'], poi['price'],
                poi['open_time'], poi['close_time'], poi['avg_stay_minutes'],
                json.dumps(poi.get('tags', []), ensure_ascii=False),
                json.dumps(poi.get('features', {}), ensure_ascii=False),
                poi.get('description', ''),
                json.dumps(poi.get('images', []), ensure_ascii=False),
                poi.get('created_at', ''), poi.get('updated_at', ''),
            ))

        conn.executemany(
            "INSERT INTO pois VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            poi_rows,
        )
        conn.executemany(
            "INSERT INTO cities(city, poi_count) VALUES (?, ?)",
            [(city, count) for city, count in city_counts.items()],
        )
        conn.executemany(
            "INSERT INTO districts(city, district, poi_count) VALUES (?, ?, ?)",
            [(city, district, count) for (city, district), count in district_counts.items()],
        )

        zone_rows = []
        for (city, zone), count in zone_counts.items():
            zone_meta = zone_lookup.get(city, {}).get(zone, {})
            center = zone_meta.get('center', (None, None))
            district = zone_meta.get('district', '')
            aliases = sorted(zone_aliases[(city, zone)])
            zone_rows.append((city, zone, district, center[0], center[1], count, json.dumps(aliases, ensure_ascii=False)))
        conn.executemany(
            "INSERT INTO zones(city, zone, district, center_lng, center_lat, poi_count, shard_aliases_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            zone_rows,
        )
        conn.commit()
    return db_path



def write_partitioned_json(data: list[dict], output_dir: str) -> None:
    partitioned_dir = os.path.join(output_dir, "poi_data_500k_partitioned")
    cities_dir = os.path.join(partitioned_dir, "cities")
    os.makedirs(cities_dir, exist_ok=True)

    manifest = {
        "format": "partitioned_json_v1",
        "total_count": len(data),
        "cities": {},
    }

    zone_lookup = _zone_lookup()
    pois_by_city = defaultdict(list)
    for poi in data:
        pois_by_city[poi["city"]].append(poi)

    for city, city_pois in pois_by_city.items():
        city_dir = os.path.join(cities_dir, _safe_path_name(city))
        zones_dir = os.path.join(city_dir, "zones")
        districts_dir = os.path.join(city_dir, "districts")
        os.makedirs(zones_dir, exist_ok=True)
        os.makedirs(districts_dir, exist_ok=True)

        city_all_path = os.path.join(city_dir, "all.json")
        write_json(city_pois, city_all_path)

        zones: dict[str, str] = {}
        pois_by_zone = defaultdict(list)
        for poi in city_pois:
            zone = str(poi.get("zone") or "").strip()
            if zone:
                pois_by_zone[zone].append(poi)
        for zone, zone_pois in pois_by_zone.items():
            zone_file = f"{_safe_path_name(zone)}.json"
            write_json(zone_pois, os.path.join(zones_dir, zone_file))
            zone_meta = zone_lookup.get(city, {}).get(zone, {})
            zones[zone] = {
                'file': f"cities/{_safe_path_name(city)}/zones/{zone_file}",
                'center': list(zone_meta.get('center', [])),
                'district': zone_meta.get('district', ''),
            }

        districts: dict[str, str] = {}
        pois_by_district = defaultdict(list)
        for poi in city_pois:
            district = str(poi.get("district") or "").strip()
            if district:
                pois_by_district[district].append(poi)
        for district, district_pois in pois_by_district.items():
            district_file = f"{_safe_path_name(district)}.json"
            write_json(district_pois, os.path.join(districts_dir, district_file))
            districts[district] = f"cities/{_safe_path_name(city)}/districts/{district_file}"

        manifest["cities"][city] = {
            "count": len(city_pois),
            "all_file": f"cities/{_safe_path_name(city)}/all.json",
            "zones": zones,
            "districts": districts,
        }

    write_json(manifest, os.path.join(partitioned_dir, "manifest.json"))



def main():
    total = 500000
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "data")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "poi_data_500k.csv")
    json_path = os.path.join(output_dir, "poi_data_500k.json")

    print(f"开始生成 {total:,} 条POI数据...")
    print(f"输出目录: {output_dir}")
    print("-" * 60)

    start_time = time.time()

    all_pois = []
    batch_size = 10000
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch = []
        for i in range(batch_start, batch_end):
            batch.append(generate_poi(i, CITIES, CATEGORIES))
        all_pois.extend(batch)
        elapsed = time.time() - start_time
        progress = batch_end / total * 100
        print(f"  进度: {batch_end:>7,}/{total:,} ({progress:5.1f}%) | 耗时: {elapsed:.1f}s")

    gen_time = time.time() - start_time
    print("-" * 60)
    print(f"数据生成完成! 耗时: {gen_time:.1f}s")

    # 写入CSV
    print(f"\n正在写入CSV文件...")
    t = time.time()
    write_csv(all_pois, csv_path)
    csv_size = os.path.getsize(csv_path) / (1024 * 1024)
    print(f"  CSV写入完成: {csv_path} ({csv_size:.1f} MB, 耗时 {time.time()-t:.1f}s)")

    # 按城市分组写入JSON
    print(f"\n正在写入JSON文件 (按城市分组)...")
    t = time.time()
    pois_by_city = defaultdict(list)
    for poi in all_pois:
        pois_by_city[poi["city"]].append(poi)
    write_json(dict(pois_by_city), json_path)
    json_size = os.path.getsize(json_path) / (1024 * 1024)
    print(f"  JSON写入完成: {json_path} ({json_size:.1f} MB, 耗时 {time.time()-t:.1f}s)")

    print(f"\n正在写入分片JSON文件 (按城市/商圈分组)...")
    t = time.time()
    write_partitioned_json(all_pois, output_dir)
    print(f"  分片JSON写入完成: {os.path.join(output_dir, 'poi_data_500k_partitioned')} (耗时 {time.time()-t:.1f}s)")

    print(f"\n正在写入SQLite数据库...")
    t = time.time()
    db_path = write_sqlite(all_pois, output_dir)
    db_size = os.path.getsize(db_path) / (1024 * 1024)
    print(f"  SQLite写入完成: {db_path} ({db_size:.1f} MB, 耗时 {time.time()-t:.1f}s)")

    # 统计信息
    print("\n" + "=" * 60)
    print("数据统计:")
    print("=" * 60)

    # 城市分布
    city_counts = {}
    cat_counts = {}
    zone_counts = defaultdict(int)
    zone_poi_count = 0
    for poi in all_pois:
        city_counts[poi["city"]] = city_counts.get(poi["city"], 0) + 1
        cat_counts[poi["category"]] = cat_counts.get(poi["category"], 0) + 1
        if poi.get("zone"):
            zone_counts[poi["zone"]] += 1
            zone_poi_count += 1

    print(f"\n城市分布 (Top 10):")
    for city, count in sorted(city_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {city}: {count:>6,} ({count/total*100:.1f}%)")

    print(f"\n商圈分布:")
    print(f"  落入商圈的POI: {zone_poi_count:,} ({zone_poi_count/total*100:.1f}%)")
    print(f"  散落城市其他区域: {total - zone_poi_count:,} ({(total - zone_poi_count)/total*100:.1f}%)")
    print(f"  商圈总数: {len(zone_counts)}")
    print(f"  Top 15 商圈:")
    for zone, count in sorted(zone_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"    {zone}: {count:>5,}")

    print(f"\n类别分布:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count:>6,} ({count/total*100:.1f}%)")

    # 评分统计
    ratings = [p["rating"] for p in all_pois]
    avg_rating = sum(ratings) / len(ratings)
    print(f"\n评分统计:")
    print(f"  平均评分: {avg_rating:.2f}")
    print(f"  最低评分: {min(ratings)}")
    print(f"  最高评分: {max(ratings)}")

    # 消费统计
    costs = [p["price"] for p in all_pois]
    avg_cost = sum(costs) / len(costs)
    print(f"\n消费统计:")
    print(f"  平均人均消费: {avg_cost:.0f}元")
    print(f"  最低人均消费: {min(costs):.0f}元")
    print(f"  最高人均消费: {max(costs):.0f}元")

    total_time = time.time() - start_time
    print(f"\n总耗时: {total_time:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
