"""Core recursive NDPI prediction function.

Pass a trained model and history/target-year arrays shaped (time, features).
Normalize NDPI as (NDPI + 1) / 2 before calling. Optional remaining columns are
known seasonal features. Output rows represent observation cutoffs; values are
returned in NDPI units. Model training and weights are not included in this preview.
"""
import numpy as np


def rolling_predict(model, train_data, test_data, time_step, n_features, n_points):
    """滚动预测，生成 n_points × n_points 的结果矩阵。"""
    results = np.zeros((n_points, n_points))

    for obs_count in range(n_points):
        if obs_count == 0:
            history = train_data[-time_step:, :].copy()
        elif obs_count < time_step:
            history = np.vstack([
                train_data[-(time_step - obs_count):, :],
                test_data[:obs_count, :]
            ])
        else:
            history = test_data[obs_count - time_step:obs_count, :].copy()

        results[obs_count, :obs_count] = (test_data[:obs_count, 0] * 2) - 1

        temp_history = history.copy()
        predictions = []

        for i in range(n_points - obs_count):
            input_data = temp_history[-time_step:, :].reshape(1, time_step, n_features)
            pred = model.predict(input_data, verbose=0)[0, 0]
            predictions.append(pred)

            current_idx = obs_count + i
            if n_features > 1:
                next_features = test_data[current_idx, 1:]
                new_row = np.concatenate(([pred], next_features)).reshape(1, n_features)
            else:
                new_row = np.array([[pred]])

            temp_history = np.vstack([temp_history, new_row])

        results[obs_count, obs_count:] = (np.array(predictions) * 2) - 1

    return results
