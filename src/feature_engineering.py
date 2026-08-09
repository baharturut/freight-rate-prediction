import numpy as np

def haversine_distance(lat1, lon1, lat2, lon2):

    R = 3958.8 # The radius of the Earth in miles.
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    distance = R * c
    return distance

def route_feature_engineering(data):
    data = data.copy()

    coord_columns = ['pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon']

    if all(col in data.columns for col in coord_columns):
        data['geo_distance'] = haversine_distance(
            data['pickup_lat'], data['pickup_lon'], 
            data['delivery_lat'], data['delivery_lon']
        )
    elif 'distance' in data.columns:
        # If the coordinates are not available:
        data['geo_distance'] = data['distance']
    else:
        data['geo_distance'] = 0.0

    # calculate route complexity
    if 'distance' in data.columns and 'geo_distance' in data.columns:
        data['route_complexity'] = data['distance'] / (data['geo_distance'] + 1e-5)
    else:
        data['route_complexity'] = 1.0

    # 3. route id creation
    if 'pickup' in data.columns and 'delivery' in data.columns:
        data['route_id'] = data['pickup'].astype(str) + '_' + data['delivery'].astype(str)

    return data

def time_seasonality_features(data):
    data = data.copy()
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    data['dayofweek'] = data['date'].dt.dayofweek
    data['is_weekend'] = data['dayofweek'].isin([5, 6]).astype(int)
    data['is_month_end'] = data['date'].dt.is_month_end.astype(int)
    data['quarter'] = data['date'].dt.quarter
    return data

def interactions_cost_features(data):
    data = data.copy()
    data['distance_x_market'] = data['distance'] * data['market_index']

    if 'quote_signal' in data.columns:
        data['distance_x_signal'] = data['distance'] * data['quote_signal']

    data['weight_per_mile'] = data['weight'] / (data['distance'] + 1e-5)

    data['equipment_avg_weight'] = data.groupby('equipment')['weight'].transform('mean')

    return data


def feature_engineering(data):  
    
    data = route_feature_engineering(data)
    data = time_seasonality_features(data)
    data = interactions_cost_features(data)
    return data