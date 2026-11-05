# Arma futbolistas_2024_25.csv a partir de las tablas de transfermarkt-datasets
# Rendimiento de la temporada 2024/25 y valor de mercado publicado despues de esa temporada
# Antes hay que bajar players, appearances, player_valuations y competitions (.csv.gz) a una carpeta
# Uso: python armar_dataset.py carpeta_con_los_archivos/

import sys
import pandas as pd

BASE = sys.argv[1] if len(sys.argv) > 1 else "datos_crudos/"

jugadores = pd.read_csv(BASE + "players.csv.gz")
apariciones = pd.read_csv(BASE + "appearances.csv.gz", parse_dates=["date"])
valores = pd.read_csv(BASE + "player_valuations.csv.gz", parse_dates=["date"])
competencias = pd.read_csv(BASE + "competitions.csv.gz")

temporada = apariciones[(apariciones["date"] >= "2024-07-01") & (apariciones["date"] <= "2025-06-30")]

ligas = {
    "GB1": "Premier League", "ES1": "LaLiga", "IT1": "Serie A", "L1": "Bundesliga", "FR1": "Ligue 1",
    "PO1": "Liga Portugal", "NL1": "Eredivisie", "TR1": "Super Lig", "BE1": "Pro League", "RU1": "Premier Liga Rusia",
    "UKR1": "Premier Liga Ucrania", "SC1": "Premiership Escocia", "GR1": "Super League Grecia", "DK1": "Superliga Dinamarca",
}

# liga principal de cada jugador, donde jugo mas minutos
en_liga = temporada[temporada["competition_id"].isin(ligas)]
minutos_liga = en_liga.groupby(["player_id", "competition_id"])["minutes_played"].sum().reset_index()
liga = minutos_liga.sort_values("minutes_played").groupby("player_id").last()["competition_id"].map(ligas).rename("liga")

# torneos europeos de clubes
nombres = competencias.set_index("competition_id")["name"]
europa = [c for c in temporada["competition_id"].unique()
          if any(t in str(nombres.get(c, "")) for t in ["champions-league", "europa-league", "conference-league"])]
print("Torneos europeos:", {c: nombres[c] for c in europa})
minutos_europa = temporada[temporada["competition_id"].isin(europa)].groupby("player_id")["minutes_played"].sum().rename("minutos_europa")

totales = temporada.groupby("player_id").agg(
    partidos=("game_id", "nunique"), minutos=("minutes_played", "sum"), goles=("goals", "sum"),
    asistencias=("assists", "sum"), amarillas=("yellow_cards", "sum"), rojas=("red_cards", "sum"))

tabla = totales.join(liga, how="inner").join(minutos_europa).reset_index()
tabla["minutos_europa"] = tabla["minutos_europa"].fillna(0).astype(int)

perfil = jugadores[["player_id", "name", "date_of_birth", "position", "sub_position", "foot", "height_in_cm", "country_of_citizenship"]]
tabla = tabla.merge(perfil, on="player_id", how="left")

# primer valor publicado despues de la temporada, de julio a diciembre de 2025
despues = valores[(valores["date"] >= "2025-07-01") & (valores["date"] <= "2025-12-31")]
valor = despues.sort_values("date").groupby("player_id").first()["market_value_in_eur"].rename("valor_eur")
tabla = tabla.merge(valor, left_on="player_id", right_index=True, how="left")

tabla = tabla.rename(columns={
    "player_id": "id_jugador", "name": "nombre", "date_of_birth": "fecha_nacimiento", "position": "posicion",
    "sub_position": "sub_posicion", "foot": "pie", "height_in_cm": "altura_cm", "country_of_citizenship": "nacionalidad"})

columnas = ["id_jugador", "nombre", "fecha_nacimiento", "nacionalidad", "liga", "posicion", "sub_posicion", "pie", "altura_cm",
            "partidos", "minutos", "goles", "asistencias", "amarillas", "rojas", "minutos_europa", "valor_eur"]

tabla[columnas].to_csv("futbolistas_2024_25.csv", index=False, encoding="utf-8")
print(tabla[columnas].shape)
