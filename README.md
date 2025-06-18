# 🤖 Automated Trading System

Docker化された仮想通貨自動取引システム（ローカル環境専用）

## 📋 システム概要

### アーキテクチャ

```
                ┌──────────────────────┐
                │   Streamlit UI       │  localhost:8501
                └────────▲─────────────┘
                         │ (HTTP)
┌────────────────────────┴─────────────────────────┐
│                    app container                 │
│  main.py  ─ CLI / Scheduler / Live Trader        │
│  core/    ─ Backtest & DataLoader (ccxt)         │
│  strategies/ ─ ユーザ実装アルゴ                  │
│  db/      ─ psql client / migrations             │
└─────────────────▲───────────────────────────────┘
                  │ (TCP 5432)
┌─────────────────┴───────────────────────────────┐
│                postgres container               │
│  DB: backtests, strategies, orders, candles     │
└──────────────────────────────────────────────────┘
```

### 主要機能

- **データ取得**: Bybit API からヒストリカル・ライブデータを取得
- **バックテスト**: pandas ベースの高速バックテストエンジン
- **ライブ取引**: リアルタイム取引実行（Bybit 対応）
- **戦略管理**: カスタム取引戦略の実装・管理
- **Web UI**: Streamlit ベースの直感的なダッシュボード
- **データベース**: PostgreSQL による永続化

## 🚀 セットアップ手順

### 1. プロジェクトの準備

```bash
git clone <repository-url>
cd auto_trader
```

### 2. API キーの設定

```bash
cp .env.example .env
```

`.env` ファイルを編集して Bybit API キーを設定:

```bash
BYBIT_API_KEY=your_api_key_here
BYBIT_API_SECRET=your_api_secret_here
```

### 3. Docker でシステムを起動

```bash
docker compose --env-file .env up --build
```

### 4. ブラウザでアクセス

```
http://localhost:8501
```

## 📖 使用方法

### CLI コマンド

#### データダウンロード
```bash
python main.py download BTCUSDT 2024-01-01 2024-04-01 1m
```

#### バックテスト実行
```bash
python main.py backtest strategies/sample_meanrev.py BTCUSDT 2024-03-01 2024-04-01
```

#### ライブ取引開始
```bash
python main.py live strategies/sample_meanrev.py BTCUSDT --qty 0.01
```

#### システム状態確認
```bash
python main.py status
```

#### バックテスト履歴表示
```bash
python main.py history
```

### Web UI の使用

1. **Dashboard**: 全体的な統計情報とサマリー
2. **Strategies**: 戦略の管理と実行
3. **Backtests**: バックテスト結果の閲覧と分析
4. **Live Trading**: ライブ取引の開始と監視
5. **Data Management**: ヒストリカルデータの管理
6. **Settings**: システム設定と状態確認

## 🔧 戦略の作成

### サンプル戦略

`strategies/sample_meanrev.py` にサンプル戦略が含まれています：

- **MeanReversionStrategy**: RSI と移動平均を使った平均回帰戦略
- **BollingerBandsMeanReversion**: ボリンジャーバンドを使った戦略

### カスタム戦略の作成

```python
import pandas as pd
import numpy as np
from typing import Dict, Any

class MyStrategy:
    def __init__(self, parameters: Dict[str, Any] = None):
        self.params = {
            'param1': 20,
            'param2': 0.02
        }
        if parameters:
            self.params.update(parameters)
        
        self.name = "My Strategy"
        self.version = "1.0"
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series with signals: 1 for buy, -1 for sell, 0 for hold
        """
        signals = pd.Series(0, index=data.index)
        
        # Your strategy logic here
        # Return 1 for buy, -1 for sell, 0 for hold
        
        return signals
    
    def get_parameters(self) -> Dict[str, Any]:
        return self.params.copy()
    
    def set_parameters(self, parameters: Dict[str, Any]):
        self.params.update(parameters)
```

## 📊 データベーススキーマ

### 主要テーブル

- **strategies**: 戦略の定義と パラメータ
- **backtests**: バックテスト結果と パフォーマンス指標
- **live_orders**: ライブ取引の注文履歴
- **candles**: ローソク足データ（OHLCV）

### サンプルクエリ

```sql
-- 最新のバックテスト結果
SELECT s.name, b.symbol, b.metrics->>'total_return' as return_pct
FROM backtests b 
JOIN strategies s ON b.strategy_id = s.id 
ORDER BY b.created_at DESC 
LIMIT 10;

-- パフォーマンスの良い戦略
SELECT s.name, 
       AVG((b.metrics->>'total_return')::float) as avg_return,
       AVG((b.metrics->>'sharpe_ratio')::float) as avg_sharpe
FROM backtests b 
JOIN strategies s ON b.strategy_id = s.id 
GROUP BY s.name 
ORDER BY avg_return DESC;
```

## ⚠️ 注意事項

### リスク管理

- **デモ取引**: 本番取引前に必ずペーパートレードでテスト
- **資金管理**: 失っても良い資金のみで取引
- **ストップロス**: 必ずリスク管理設定を行う
- **監視**: ライブ取引中は定期的にシステムを監視

### セキュリティ

- API キーは `.env` ファイルで管理し、絶対に公開しない
- 本番環境では適切なファイアウォール設定を行う
- 定期的にパスワードと API キーを更新する

## 🛠️ 開発・カスタマイズ

### 開発環境

```bash
# 開発用にローカルで実行
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# データベース初期化
python main.py init

# Streamlit UI のみ起動
streamlit run ui.py
```

### 拡張のヒント

1. **TimescaleDB**: 時系列データの高速化
2. **Optuna**: 自動パラメータ最適化
3. **Prometheus**: システム監視
4. **Redis**: 高速キャッシュ
5. **WebSocket**: リアルタイムデータ配信

### テスト

```bash
# 基本的なシステムテスト
python main.py status

# バックテスト例
python main.py backtest strategies/sample_meanrev.py BTCUSDT 2024-01-01 2024-02-01

# データダウンロードテスト
python main.py download BTCUSDT 2024-01-01 2024-01-02 1m
```

## 📈 パフォーマンス最適化

### バックテスト高速化

- Numba JIT コンパイルによる数値計算の最適化
- ベクトル化された pandas 操作
- メモリ効率的なデータ処理

### データベース最適化

- 適切なインデックス設定
- パーティショニング（大量データ対応）
- 定期的な VACUUM と統計更新

## 🆘 トラブルシューティング

### よくある問題

1. **API エラー**: API キーの設定を確認
2. **データベース接続エラー**: PostgreSQL コンテナの状態を確認
3. **メモリ不足**: Docker のメモリ制限を増加
4. **ポートエラー**: 5432, 8501 ポートが使用可能か確認

### ログの確認

```bash
# アプリケーションログ
docker compose logs app

# データベースログ
docker compose logs db

# リアルタイムログ監視
docker compose logs -f
```

## 📄 ライセンス

このプロジェクトは教育・研究目的のものです。実際の取引での使用は自己責任で行ってください。

## 🤝 コントリビューション

プルリクエストやイシューの報告を歓迎します。新機能の提案や バグ報告は GitHub Issues をご利用ください。

## 📞 サポート

- ドキュメントを確認してから質問してください
- 具体的なエラーメッセージを含めて報告してください
- 再現手順を明確に記載してください