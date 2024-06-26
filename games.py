import pandas as pd
import optuna
from hyperparameter_tunning import objective_log, objective_rf, objective_xgb
from data_preprocessing import preprcessing_classification,train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

def group_arrangemnt(group_teams):
    left_pointer= 0
    right_pointer=1 
    fixture_arrangemnt= True
    group_games=[]
    while fixture_arrangemnt:
        home_team= group_teams[left_pointer]
        away_team= group_teams[right_pointer]
        right_pointer +=1
        game= [home_team, away_team]
        group_games.append(game)
        if right_pointer > (len(group_teams)-1):
            left_pointer +=1
            if left_pointer == len(group_teams)-1:
                fixture_arrangemnt= False
            else:
                right_pointer= left_pointer+1
            
    return group_games

def game_preprocessing(team1, team2, copa_america_rankings,country_encoder):
    if team1 == 'United States':
        home_team= team1
        away_team= team2
        neutral= False
    elif team2 == 'United States':
        home_team= team2
        away_team= team1
        neutral= False
    else:
        home_team= team1
        away_team= team2
        neutral= True
    
    rank_away_team= copa_america_rankings[copa_america_rankings['country_full'] == away_team]['rank'].values[0]
    rank_home_team= copa_america_rankings[copa_america_rankings['country_full'] == home_team]['rank'].values[0]
    total_points_away_team= copa_america_rankings[copa_america_rankings['country_full'] == away_team]['total_points'].values[0]
    total_points_home_team= copa_america_rankings[copa_america_rankings['country_full'] == home_team]['total_points'].values[0]
    game_data={'home_team': home_team,
      'away_team': away_team,
      'rank_home_team':rank_home_team,
      'rank_away_team': rank_away_team,
      'total_points_away_team': total_points_away_team,
      'total_points_home_team': total_points_home_team,
      'neutral': neutral,
      'ranking_diff': abs(rank_away_team-rank_home_team),
      'total_point_difference': abs(total_points_away_team-total_points_home_team)}   
    
    test_predticion= pd.DataFrame([game_data])
    encoded_away_team= country_encoder.transform(test_predticion[['away_team']].values)
    encoded_home_team= country_encoder.transform(test_predticion[['home_team']].values)
    away_team_df= pd.DataFrame(encoded_away_team,columns= country_encoder.get_feature_names_out(['away_team']))
    home_team_df= pd.DataFrame(encoded_home_team,columns= country_encoder.get_feature_names_out(['home_team']))
    df_targets= pd.concat([away_team_df, home_team_df, test_predticion.drop(columns=['away_team', 'home_team','neutral'])], axis=1)
    return df_targets, test_predticion['away_team'].values[0], test_predticion['home_team'].values[0]

def previous_games_splits(team1, team2, copa_america_df, label_encoder_countries):
    if team1 == 'United States' or team2== 'United States':
        game_df = copa_america_df[
        ((copa_america_df['home_team'].isin([team1, team2])) | 
        (copa_america_df['away_team'].isin([team1, team2]))) 
        ].copy()
    else:
        game_df = copa_america_df[
        ((copa_america_df['home_team'].isin([team1, team2])) | 
        (copa_america_df['away_team'].isin([team1, team2])))
        ].copy()
    
    # if team1 == 'United States' or team2== 'United States':
    #     game_df_team1 = copa_america_df[
    #     ((copa_america_df['home_team']== team1) | 
    #     (copa_america_df['away_team'] == team1)) 
    #     ].copy()
    #     game_df_team1= game_df_team1.iloc[-40:]
    #     game_df_team2 = copa_america_df[
    #     ((copa_america_df['home_team'] == team2) | 
    #     (copa_america_df['away_team']== team2))
    #     ].copy()
    #     game_df_team2= game_df_team2.iloc[-40:]
        
    # else:
    #     game_df_team1 = copa_america_df[
    #     ((copa_america_df['home_team']== team1) | 
    #     (copa_america_df['away_team'] == team1)) & 
    #     (copa_america_df['neutral'] == True)
    #     ].copy()
    #     game_df_team1= game_df_team1.iloc[-40:]
    #     game_df_team2 = copa_america_df[
    #     ((copa_america_df['home_team']== team2) | 
    #     (copa_america_df['away_team'] == team2)) & 
    #     (copa_america_df['neutral'] == True)
    #     ].copy()
    #     game_df_team2= game_df_team2.iloc[-40:]
        
    
    # game_df= pd.concat([game_df_team1, game_df_team2], axis=0)
    game_df.sort_values(by='date', ascending=True, inplace=True)
    game_df.reset_index(drop=True, inplace= True)
    
    game_features, game_targets= preprcessing_classification(game_df,label_encoder_countries)
    X_train,y_train, X_val, y_val= train_test_split(game_features,game_targets,train_per=0.6, test_set=False)
    return X_train, y_train, X_val, y_val

def target_encoder_games(y_train, y_val):
    label_encoder_target= LabelEncoder()
    label_encoder_target.fit(y_train)
    y_train= label_encoder_target.transform(y_train)
    y_val= label_encoder_target.transform(y_val)
    return y_train, y_val

def prediction_game(X_train,y_train,X_val,y_val,game_data,home_team, away_team):
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)
    y_train_enc, y_val_enc= target_encoder_games(y_train, y_val)
    study_log= optuna.create_study(direction= 'maximize')
    study_log.optimize(lambda trial: objective_log(trial, X_train,y_train,X_val,y_val), n_trials= 100)
    accuracy_log= study_log.best_value
    
    study_rf= optuna.create_study(direction= 'maximize')
    study_rf.optimize(lambda trial: objective_rf(trial, X_train,y_train,X_val,y_val), n_trials= 100)
    accuracy_rf= study_rf.best_value
    
    study_xgb= optuna.create_study(direction= 'maximize')
    study_xgb.optimize(lambda trial: objective_xgb(trial, X_train,y_train_enc,X_val,y_val_enc), n_trials= 100)
    accuracy_xgb= study_xgb.best_value
    
    dict_accuracy_scores={'XgBoost': accuracy_xgb,
                      'Random Forest':accuracy_rf,
                      'Logistic Regression': accuracy_log}

    bar_accuraies= px.bar(x= dict_accuracy_scores.keys(),
                      y= dict_accuracy_scores.values(),
                      color= dict_accuracy_scores.keys(),
                      color_continuous_scale='Viridis',
                      title= 'Accuracies Per Model')
    bar_accuraies.update_layout(xaxis_title='Model',
                            yaxis_title='Accuracy')
    bar_accuraies.show()
    
    if accuracy_log > accuracy_xgb and accuracy_log > accuracy_rf:
        best_model= LogisticRegression(solver='lbfgs', max_iter= study_log.best_params['max_iter'], C= study_log.best_params['C'], class_weight='balanced')
        best_model.fit(X_train, y_train)
        y_pred= best_model.predict(game_data)
        probabilities_classes= best_model.predict_proba(game_data)
        classes= best_model.classes_
        classes[0]= away_team
        classes[1]= home_team
        prob_data_frame= pd.DataFrame(probabilities_classes, columns= classes)
        # prob_data_frame= prob_data_frame.rename(columns= {classes[0]:away_team,
        #                                                   classes[1]: home_team})
        if y_pred == 'away_team':
            y_pred_team= away_team
        elif y_pred == 'home_team':
            y_pred_team = home_team
        else:
            y_pred_team='tied'
        return y_pred_team, prob_data_frame
        
    elif accuracy_rf > accuracy_xgb and accuracy_rf > accuracy_log:
        best_model= RandomForestClassifier(n_estimators= study_rf.best_params['n_estimators'], 
                                  max_depth=study_rf.best_params['max_depth'], 
                                  min_samples_split= study_rf.best_params['min_samples_split'], 
                                  min_samples_leaf=study_rf.best_params['min_samples_leaf'],
                                  random_state=42)

        best_model.fit(X_train, y_train)
        y_pred= best_model.predict(game_data)
        probabilities_classes= best_model.predict_proba(game_data)
        classes= best_model.classes_
        classes[0]= away_team
        classes[1]= home_team
        prob_data_frame= pd.DataFrame(probabilities_classes, columns= classes)
        # prob_data_frame= prob_data_frame.rename(columns= {classes[0]:away_team,
        #                                                   classes[1]: home_team})
        if y_pred == 'away_team':
            y_pred_team= away_team
        elif y_pred == 'home_team':
            y_pred_team = home_team
        else:
            y_pred_team='tied'
        return y_pred_team, prob_data_frame
    
    elif accuracy_xgb > accuracy_rf and accuracy_xgb > accuracy_log:
        best_model= XGBClassifier(objective='multi:softmax', num_class= 3, 
                             n_estimators= study_xgb.best_params['n_estimators'], 
                             learning_rate=study_xgb.best_params['learning_rate'], 
                             max_depth= study_xgb.best_params['max_depth'], 
                             random_state=42)
        best_model.fit(X_train, y_train_enc)
        y_pred= best_model.predict(game_data)
        probabilities_classes= best_model.predict_proba(game_data)
        classes= best_model.classes_
        prob_data_frame= pd.DataFrame(probabilities_classes, columns= classes)
        prob_data_frame= prob_data_frame.rename(columns= {classes[0]:away_team,
                                                          classes[1]: home_team,
                                                          classes[2]: 'tied'})
        if y_pred == 0:
            y_pred_team= away_team
        elif y_pred == 1:
            y_pred_team = home_team
        else:
            y_pred_team='tied'
        return y_pred_team, prob_data_frame
    
    else:
        best_model= RandomForestClassifier(n_estimators= study_rf.best_params['n_estimators'], 
                                  max_depth=study_rf.best_params['max_depth'], 
                                  min_samples_split= study_rf.best_params['min_samples_split'], 
                                  min_samples_leaf=study_rf.best_params['min_samples_leaf'],
                                  random_state=42)

        best_model.fit(X_train, y_train)
        y_pred= best_model.predict(game_data)
        probabilities_classes= best_model.predict_proba(game_data)
        classes= best_model.classes_
        classes[0]= away_team
        classes[1]= home_team
        prob_data_frame= pd.DataFrame(probabilities_classes, columns= classes)
        # prob_data_frame= prob_data_frame.rename(columns= {classes[0]:away_team,
        #                                                    classes[1]: home_team})
        if y_pred == 'away_team':
            y_pred_team= away_team
        elif y_pred == 'home_team':
            y_pred_team = home_team
        else:
            y_pred_team='tied'
        return y_pred_team, prob_data_frame

def pie_chart(prob_data_frame, home_team, away_team):
    df_data_frame_melt= prob_data_frame.melt(var_name='Team', value_name='Probability')
    fig= px.pie(labels= df_data_frame_melt['Team'], values= df_data_frame_melt['Probability'], title=f'Probabilities of {home_team} vs {away_team}', names= df_data_frame_melt['Team'])
    fig.update_traces(textposition='inside', textfont_size= 15, textinfo= 'percent')
    fig.show() 


def group_simulation(group, group_games,country_encoder, final_copa_america_df, copa_america_rankings):
    group_games= group_arrangemnt(group)
    list_points=[0,0,0,0]
    for game in group_games:
        home_team= game[0]
        home_team_idx= group.index(home_team)
        away_team= game[1]
        away_team_idx= group.index(away_team)
        game_data,away_team, home_team= game_preprocessing(home_team, away_team, copa_america_rankings,country_encoder)
        X_train,y_train,X_val, y_val= previous_games_splits(home_team,away_team, final_copa_america_df,country_encoder)
        
        y_pred_team, prob_data_frame= prediction_game(X_train,y_train, X_val, y_val, game_data,away_team=away_team, home_team=home_team)
        pie_chart(prob_data_frame, home_team, away_team)
        if y_pred_team == home_team:
            list_points[home_team_idx]+=3
        elif y_pred_team == away_team:
            list_points[away_team_idx] += 3 
        else:
            list_points[home_team_idx]+=1
            list_points[away_team_idx]+=1
    group_table= pd.DataFrame({
        "Teams": group,
        "Points": list_points
    })
    return group_table
    

def classify_teams(group_table):
    sorted_table=group_table.sort_values(by='Points', ascending=False)
    first_place= sorted_table['Teams'].values[0]
    second_place= sorted_table['Teams'].values[1]
    return first_place, second_place


def quaterfinals_games(groupA_table, groupB_table, groupC_table, groupD_table):
    groupA_first_team, groupA_second_team= classify_teams(groupA_table)
    groupB_first_team, groupB_second_team= classify_teams(groupB_table)
    groupC_first_team, groupC_second_team= classify_teams(groupC_table)
    groupD_first_team, groupD_second_team= classify_teams(groupD_table)
    
    quaterfinals_games=[[groupA_first_team,groupB_second_team],[groupB_first_team, groupA_second_team], [groupC_first_team, groupD_second_team],[groupD_first_team, groupC_second_team]]
    return quaterfinals_games

def knockout_predictions(stage_games,country_encoder, final_copa_america_df, copa_america_rankings):
    list_winners=[]
    for game in stage_games:
        game_data,away_team, home_team= game_preprocessing(game[0], game[1], copa_america_rankings, country_encoder)
        X_train,y_train,X_val, y_val= previous_games_splits(home_team,away_team, final_copa_america_df, country_encoder)
        
        y_pred_team, prob_data_frame= prediction_game(X_train,y_train, X_val, y_val, game_data,away_team=away_team, home_team=home_team)
        print(prob_data_frame)
        pie_chart(prob_data_frame, home_team, away_team)
        if y_pred_team == home_team:
            list_winners.append(home_team)
        elif y_pred_team == away_team:
            list_winners.append(away_team)
        else:
            list_winners.append('tied')
    return list_winners

def semifinals_games(semifinals_teams):
    games=[[semifinals_teams[0],semifinals_teams[1]],[semifinals_teams[2],semifinals_teams[3]]]
    return games