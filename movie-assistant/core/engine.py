"""
Движок рекомендаций фильмов.
Реализует 4 режима, требуемых заданием:
  - топ популярных (взвешенный рейтинг)
  - топ внутри жанра
  - похожие по контенту (жанры + десятилетие)
  - похожие по коллаборативной фильтрации (item-based)

"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MIN_RATINGS_FOR_ITEM = 8


class MovieEngine:
    def __init__(self, data_dir: Path | str | None = None):
        root = Path(data_dir) if data_dir else DATA_DIR
        self._load(root)
        self._prepare_popularity()
        self._prepare_content()
        self._prepare_collab()

    # load
    def _load(self, root: Path):
        movies = pd.read_csv(root / "movies.csv")
        ratings = pd.read_csv(root / "ratings.csv")

        movies = movies.rename(columns={
            "movieId": "id",
            "imdbId": "imdb",
            "tmdbId": "tmdb",
        })
        if "year" not in movies.columns:
            years = movies["title"].str.extract(r"\((\d{4})\)\s*$", expand=False)
            movies["year"] = pd.to_numeric(years, errors="coerce")

        movies["genres"] = movies["genres"].fillna("").replace("(no genres listed)", "")
        movies["year"] = movies["year"].astype("Int64")

        ratings = ratings.rename(columns={
            "userId": "user",
            "movieId": "id",
            "rating": "score",
        })
        ratings = ratings.drop_duplicates(["user", "id"])
        ratings = ratings[ratings["id"].isin(movies["id"])]

        self.movies = movies.set_index("id", drop=False)
        self.ratings = ratings

    # popularity
    def _prepare_popularity(self):
        stats = (
            self.ratings.groupby("id")
            .agg(mean_score=("score", "mean"), votes=("score", "count"))
            .reindex(self.movies.index)
        )
        stats["votes"] = stats["votes"].fillna(0).astype(int)
        stats["mean_score"] = stats["mean_score"].fillna(0.0)

        v = stats["votes"]
        R = stats["mean_score"]
        m = float(v.quantile(0.90))
        C = float(R[v > 0].mean())
        stats["weight"] = (v * R + m * C) / (v + m)
        """
            v — сколько человек оценили этот фильм
            R — средняя оценка фильма
            m — порог доверия (сколько голосов считаем достаточным)
            C — среднее по всем фильмам
        """
        stats["weight"] = stats["weight"].fillna(0.0)

        self.catalog = self.movies.join(stats)
        self._m = m
        self._C = C

    def top_popular(self, limit: int = 10) -> list[dict]:
        frame = self.catalog[self.catalog["votes"] > 0]
        frame = frame.sort_values("weight", ascending=False).head(limit)
        return self._rows(frame)

    def top_by_genre(self, genre: str, limit: int = 10) -> list[dict]:
        genre = (genre or "").strip()
        if not genre:
            raise ValueError("Укажите жанр")
        mask = self.catalog["genres"].str.split("|").apply(
            lambda parts: genre in parts if isinstance(parts, list) else False
        )
        frame = self.catalog[mask & (self.catalog["votes"] > 0)]
        if frame.empty:
            raise ValueError(f"Жанр «{genre}» не найден или нет оценённых фильмов")
        frame = frame.sort_values("weight", ascending=False).head(limit)
        return self._rows(frame)

    def available_genres(self) -> list[str]:
        bag: set[str] = set()
        for raw in self.catalog["genres"]:
            bag.update(g for g in raw.split("|") if g)
        return sorted(bag)

    # content
    def _prepare_content(self):
        """Бинарные признаки: каждый жанр + десятилетие."""
        genres = self.available_genres()
        decades = sorted({
            int(y) // 10 * 10
            for y in self.catalog["year"].dropna().astype(int)
        })
        self._feat_names = genres + [f"d{d}" for d in decades]
        name_to_col = {n: i for i, n in enumerate(self._feat_names)}

        n = len(self.catalog)
        mat = np.zeros((n, len(self._feat_names)), dtype=np.float32)

        for row_i, (_, rec) in enumerate(self.catalog.iterrows()):
            for g in rec["genres"].split("|"):
                if g in name_to_col:
                    mat[row_i, name_to_col[g]] = 1.0
            if pd.notna(rec["year"]):
                key = f"d{int(rec['year']) // 10 * 10}"
                if key in name_to_col:
                    mat[row_i, name_to_col[key]] = 1.0

        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._content_mat = mat / norms
        self._pos = {int(i): n for n, i in enumerate(self.catalog.index)}

    def similar_by_content(self, title_query: str, limit: int = 10) -> list[dict]:
        movie_id = self._resolve_title(title_query)
        pos = self._pos[movie_id]
        vec = self._content_mat[pos]
        sims = self._content_mat @ vec
        sims[pos] = -1.0

        votes = self.catalog["votes"].values
        order = np.argsort(-sims)
        chosen = []
        for idx in order:
            if votes[idx] < 3:
                continue
            chosen.append(idx)
            if len(chosen) >= limit:
                break

        frame = self.catalog.iloc[chosen].copy()
        frame["similarity"] = sims[chosen]
        return self._rows(frame, extra="similarity")

    # collab
    def _prepare_collab(self):
        users = np.sort(self.ratings["user"].unique())
        items = np.sort(self.ratings["id"].unique())
        u_map = {int(u): i for i, u in enumerate(users)}
        i_map = {int(m): i for i, m in enumerate(items)}

        rows = self.ratings["user"].map(u_map).values
        cols = self.ratings["id"].map(i_map).values
        data = self.ratings["score"].values.astype(np.float32)
        user_item = csr_matrix(
            (data, (rows, cols)), shape=(len(users), len(items))
        )

        counts = np.asarray((user_item > 0).sum(axis=0)).ravel()
        keep = np.where(counts >= MIN_RATINGS_FOR_ITEM)[0]
        self._item_ids = items[keep]
        self._item_pos = {int(m): n for n, m in enumerate(self._item_ids)}

        item_user = user_item[:, keep].T.tocsr().astype(np.float32)
        sq = item_user.multiply(item_user).sum(axis=1)
        norms = np.sqrt(np.asarray(sq).ravel())
        norms[norms == 0] = 1.0
        inv = 1.0 / norms
        item_user = item_user.multiply(inv[:, np.newaxis]).tocsr()
        self._item_user = item_user

    def similar_by_collab(self, title_query: str, limit: int = 10) -> list[dict]:
        movie_id = self._resolve_title(title_query)
        if movie_id not in self._item_pos:
            raise ValueError(
                "У фильма слишком мало оценок для коллаборативной фильтрации. "
                "Попробуйте другой фильм или режим «по контенту»."
            )
        pos = self._item_pos[movie_id]
        vec = self._item_user.getrow(pos)
        sims = (self._item_user @ vec.T).toarray().ravel()
        sims[pos] = -1.0

        order = np.argsort(-sims)[:limit]
        ids = [int(self._item_ids[j]) for j in order]
        frame = self.catalog.loc[ids].copy()
        frame["similarity"] = sims[order]
        return self._rows(frame, extra="similarity")

    # utils
    def _resolve_title(self, query: str) -> int:
        query = (query or "").strip()
        if not query:
            raise ValueError("Укажите название фильма")
        mask = self.catalog["title"].str.contains(
            re.escape(query), case=False, na=False
        )
        found = self.catalog[mask].sort_values("votes", ascending=False)
        if found.empty:
            raise ValueError(f"Фильм «{query}» не найден")
        return int(found.iloc[0]["id"])

    def _rows(self, frame: pd.DataFrame, extra: str | None = None) -> list[dict]:
        cols = ["id", "title", "year", "genres", "mean_score", "votes", "weight"]
        if extra and extra in frame.columns:
            cols.append(extra)
        out = []
        for _, row in frame[cols].iterrows():
            item = {
                "id": int(row["id"]),
                "title": row["title"],
                "year": None if pd.isna(row["year"]) else int(row["year"]),
                "genres": row["genres"],
                "mean_score": round(float(row["mean_score"]), 2),
                "votes": int(row["votes"]),
                "weight": round(float(row["weight"]), 3),
            }
            if extra and extra in row.index:
                item[extra] = round(float(row[extra]), 3)
            out.append(item)
        return out


_engine: MovieEngine | None = None


def get_engine() -> MovieEngine:
    global _engine
    if _engine is None:
        _engine = MovieEngine()
    return _engine
