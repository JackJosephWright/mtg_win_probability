import joblib
import os


class PhaseBasedModel:
    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = os.path.join(
                os.path.dirname(__file__), '..', '..', 'models', 'win_probability', 'phase_models'
            )
        self.early_model = joblib.load(os.path.join(model_dir, 'xgboost_early_game.pkl'))
        self.mid_model = joblib.load(os.path.join(model_dir, 'xgboost_mid_game.pkl'))
        self.late_model = joblib.load(os.path.join(model_dir, 'random_forest_late_game.pkl'))

    def _select_model(self, turn):
        if 1 <= turn <= 5:
            return self.early_model
        elif 6 <= turn <= 10:
            return self.mid_model
        else:
            return self.late_model

    def predict(self, data):
        data = data.drop(columns=['game_id', 'unique_id'], errors='ignore')
        turn = data['turn'].iloc[0]
        model = self._select_model(turn)
        X = data.drop(columns=['turn', 'won'], errors='ignore')
        return model.predict(X)

    def predict_proba(self, data):
        """Returns win probability (probability of class 1)."""
        data = data.drop(columns=['game_id', 'unique_id'], errors='ignore')
        turn = data['turn'].iloc[0]
        model = self._select_model(turn)
        X = data.drop(columns=['turn', 'won'], errors='ignore')
        return model.predict_proba(X)[:, 1]
