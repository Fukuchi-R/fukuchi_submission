import os
import sys
import numpy as np
import chess
import chess.pgn
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# PGNファイルのパスを取得する関数
def get_pgn_path():
    if getattr(sys, 'frozen', False):  # PyInstaller で実行ファイル化されている場合
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, "testdata", "1500-1600.pgn")

# PGNファイルを読み込む関数
def read_pgn():
    file_path = get_pgn_path()
    games = []
    with open(file_path, "r") as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            games.append(game)
    return games

# 盤面をエンコードする関数
def encode_board(board):
    encoded_board = np.zeros((8, 8, 17), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            x, y = divmod(square, 8)
            if piece.color == chess.WHITE:
                encoded_board[x, y, piece.piece_type - 1] = 1
            else:
                encoded_board[x, y, piece.piece_type + 5] = 1
    if board.has_kingside_castling_rights(chess.WHITE):
        encoded_board[0, 0, 12] = 1
    if board.has_queenside_castling_rights(chess.WHITE):
        encoded_board[0, 1, 13] = 1
    if board.has_kingside_castling_rights(chess.BLACK):
        encoded_board[0, 0, 14] = 1
    if board.has_queenside_castling_rights(chess.BLACK):
        encoded_board[0, 1, 15] = 1
    if board.turn == chess.WHITE:
        encoded_board[0, 0, 16] = 1
    else:
        encoded_board[0, 1, 16] = 1
    return encoded_board

# 駒とマスの組み合わせをまとめたラベルを作成する関数
def create_combined_label(move, board):
    piece_type = board.piece_at(move.from_square).piece_type - 1
    square_index = move.to_square
    return piece_type * 64 + square_index

# モデルを構築して学習する関数
def build_and_train_combined_model(output_size, data, labels):
    board_inputs = tf.keras.Input(shape=(8, 8, 17))
    conv1 = layers.Conv2D(32, 3, activation='relu')(board_inputs)
    pooling1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = layers.Conv2D(64, 3, activation='relu')(pooling1)
    flatten = layers.Flatten()(conv2)
    dense1 = layers.Dense(128, activation='relu')(flatten)
    combined_output = layers.Dense(output_size, activation='softmax')(dense1)
    model = models.Model(inputs=board_inputs, outputs=combined_output, name="chess_ai_v4_combined")
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'])
    model.fit(np.array(data), np.array(labels), batch_size=32, epochs=100, verbose=0)
    return model

# メイン関数
if __name__ == "__main__":
    print("PGNファイルを読み込み中...")
    games = read_pgn()
    
    print("データを分割中...")
    train_games, test_games = train_test_split(games, test_size=0.2)
    train_inputs, train_labels_combined = [], []
    test_inputs, test_labels_combined = [], []
    test_san_moves, test_colors = [], []

    for game in train_games:
        board = game.board()
        for move in game.mainline_moves():
            encoded_board = encode_board(board)
            train_inputs.append(encoded_board)
            combined_label = create_combined_label(move, board)
            train_labels_combined.append(combined_label)
            board.push(move)

    for game in test_games:
        board = game.board()
        for move in game.mainline_moves():
            encoded_board = encode_board(board)
            test_inputs.append(encoded_board)
            combined_label = create_combined_label(move, board)
            test_labels_combined.append(combined_label)
            test_san_moves.append(board.san(move))
            test_colors.append("White" if board.turn == chess.WHITE else "Black")
            board.push(move)

    combined_output_size = 64 * 6
    print("モデルを構築して学習中...")
    model = build_and_train_combined_model(output_size=combined_output_size,
                                           data=np.array(train_inputs),
                                           labels=np.array(train_labels_combined))

    print("テストデータで評価中...")
    test_predictions = model.predict(np.array(test_inputs))
    y_pred_classes = np.argmax(test_predictions, axis=1)
    y_test_classes = np.array(test_labels_combined)

    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    print(f"Combined Prediction Accuracy: {accuracy:.4f}")

    num_moves_to_display = 50
    for i in range(min(num_moves_to_display, len(y_test_classes))):
        actual_combined_label = y_test_classes[i]
        predicted_combined_label = y_pred_classes[i]
        actual_piece_type = actual_combined_label // 64
        actual_square = actual_combined_label % 64
        predicted_piece_type = predicted_combined_label // 64
        predicted_square = predicted_combined_label % 64
        actual_move_x, actual_move_y = divmod(actual_square, 8)
        actual_file = chr(ord('a') + actual_move_y)
        actual_rank = 1 + actual_move_x
        predicted_move_x, predicted_move_y = divmod(predicted_square, 8)
        predicted_file = chr(ord('a') + predicted_move_y)
        predicted_rank = 1 + predicted_move_x
        actual_piece = chess.PIECE_NAMES[actual_piece_type + 1]
        predicted_piece = chess.PIECE_NAMES[predicted_piece_type + 1]
        actual_piece_display = actual_piece.upper() if test_colors[i] == "White" else actual_piece.lower()
        predicted_piece_display = predicted_piece.upper() if test_colors[i] == "White" else predicted_piece.lower()
        match = "〇" if actual_combined_label == predicted_combined_label else "×"
        print(f"{i+1}. {test_colors[i]} 実際の手: {test_san_moves[i]} ({actual_file}{actual_rank}, {actual_piece_display}), "
              f"予測された手: {predicted_file}{predicted_rank} ({predicted_piece_display}) 一致: {match}")
