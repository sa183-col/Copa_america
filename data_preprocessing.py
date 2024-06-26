import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def merge_rank_date(df_data, df_rank, home_away):
    copa_america_df= pd.merge(df_data, df_rank, left_on=[f'{home_away}_team'], right_on=['country_full'], how='inner')
    copa_america_df= copa_america_df[copa_america_df['date'] >= copa_america_df['rank_date']]
    copa_america_df['date_diff']= (copa_america_df['date']- copa_america_df['rank_date']).abs()
    copa_america_df= copa_america_df.sort_values(by=['date',f'{home_away}_team','date_diff'])
    copa_america_df= copa_america_df.groupby(['date',f'{home_away}_team']).first().reset_index()
    copa_america_df= copa_america_df.drop(['country_abrv','rank_date','date_diff','country_full'], axis=1)
    copa_america_df= copa_america_df.rename(columns={'rank':f'rank_{home_away}_team', 'total_points':f'total_points_{home_away}_team', 'previous_points': f'previous_points_{home_away}_team','rank_change':f'rank_change_{home_away}_team', 'confederation':f'confederation_{home_away}_team'})
    return copa_america_df
    
def full_merge_df(df_results, df_ranking, list_teams):
    data_results_copa_america_2024= df_results[(df_results['home_team'].isin(list_teams)) | (df_results['away_team'].isin(list_teams))].copy()
    df_ranking= df_ranking.dropna().copy()
    df_ranking['country_full']= df_ranking['country_full'].replace('USA', 'United States')   
    df_ranking['rank_date']= pd.to_datetime(df_ranking['rank_date'])
    copa_america_df= merge_rank_date(data_results_copa_america_2024, df_ranking,'home')
    full_copa_america_df= merge_rank_date(copa_america_df, df_ranking,'away')
    return full_copa_america_df, df_ranking

def winning_team(home_score, away_score):
    if home_score > away_score:
        return "home_team"
    
    elif home_score < away_score:
        return "away_team"
    
    else:
        return "tied"

def feature_engineering(full_data_frame):
    full_data_frame['winning_team']= full_data_frame.apply(lambda x : winning_team(x['home_score'], x['away_score']), axis= 1)
    full_data_frame['goal_difference']= (full_data_frame['home_score'] - full_data_frame['away_score']).abs()
    full_data_frame['total_point_difference']=(full_data_frame['total_points_away_team']- full_data_frame['total_points_away_team']).abs()
    full_data_frame['ranking_diff']= (full_data_frame['rank_home_team']- full_data_frame['rank_away_team']).abs()
    return full_data_frame


def preprcessing_classification(copa_data_frame,country_encoder):
    df_copa_multi_class= copa_data_frame[['date','away_team','home_team','rank_home_team','rank_away_team','total_points_away_team','total_points_home_team','winning_team','ranking_diff','total_point_difference']].copy()
    away_team_encoded= country_encoder.transform(df_copa_multi_class[['away_team']].values)
    home_team_encoded= country_encoder.transform(df_copa_multi_class[['home_team']].values)
    
    away_team_df= pd.DataFrame(away_team_encoded,columns= country_encoder.get_feature_names_out(['away_team']))
    home_team_df= pd.DataFrame(home_team_encoded,columns= country_encoder.get_feature_names_out(['home_team']))
    
    away_team_df.reset_index(drop=True, inplace= True)
    home_team_df.reset_index(drop= True, inplace= True)
    df_copa_multi_class.reset_index(drop=True, inplace= True)
    
    
    data= pd.concat([away_team_df, home_team_df, df_copa_multi_class.drop(columns=['away_team', 'home_team'])], axis=1)
    features= data.drop(['winning_team','date'], axis=1)
    target_multi_class= data['winning_team']
    return features, target_multi_class

def train_test_split(features, targets, train_per,test_set: bool):
    train_size= int(len(features)*train_per)
    X_train= features[:train_size]
    y_train= targets[:train_size]
    if test_set:
        val_size=int((len(features)*(1-train_per))/2)+1
        print(val_size)
        X_val= features[train_size:train_size + val_size]
        y_val= targets[train_size:train_size + val_size]
    
        X_test= features[train_size + val_size:]
        y_test= targets[train_size + val_size:]
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    else:
        val_size= int(len(features)*(1-train_per))
        X_val= features[train_size:]
        y_val= targets[train_size:]
        return X_train, y_train, X_val, y_val

def target_encoder(y_train, y_val,y_test):
    label_encoder_target= LabelEncoder()
    label_encoder_target.fit(y_train)
    y_train= label_encoder_target.transform(y_train)
    y_val= label_encoder_target.transform(y_val)
    y_test= label_encoder_target.transform(y_test)
    return y_train, y_val, y_test