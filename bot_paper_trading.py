"""
BOT DE TRADING IA - OR (GOLD) - MODE SIMULATION (paper trading)
=================================================================
Tourne une fois par jour via GitHub Actions.
Aucune connexion à un broker réel, aucun argent réel engagé.

Stratégie swing (échelle journalière) avec effet de levier x2 :
validé en backtest avec un meilleur ratio rendement/risque que le
simple achat-conservation (buy & hold), drawdown max ~12% vs ~25%.

⚠️ Le levier amplifie aussi les pertes. En conditions réelles (pas en
simulation), un mouvement de prix brutal peut déclencher une liquidation
avant même que le stop-loss ne s'exécute — ce risque n'est pas modélisé ici.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

FICHIER_ETAT = "etat_portefeuille.json"
FICHIER_LOG = "historique_trades.csv"

CAPITAL_DEPART = 30
TAKE_PROFIT = 0.02          # +2% sur le prix
STOP_LOSS = 0.01            # -1% sur le prix
SEUIL_CONFIANCE = 0.55
LEVIER = 2                  # effet de levier x2, validé en backtest
TAUX_INTERET_ANNUEL = 0.06  # coût d'emprunt estimé (6%/an), à ajuster selon le broker réel

FEATURES = ["retour_1", "retour_5", "retour_15", "volatilite",
            "ecart_moyennes", "heure", "volume_relatif", "rsi", "retour_dollar"]


def charger_etat():
    if os.path.exists(FICHIER_ETAT):
        with open(FICHIER_ETAT, "r") as f:
            return json.load(f)
    return {"capital": CAPITAL_DEPART, "position_ouverte": False,
            "prix_entree": None, "date_entree": None}


def sauvegarder_etat(etat):
    with open(FICHIER_ETAT, "w") as f:
        json.dump(etat, f, indent=2)


def enregistrer_trade(date, type_action, prix, capital):
    ligne = pd.DataFrame([{
        "date": date, "action": type_action, "prix": prix, "capital": capital
    }])
    if os.path.exists(FICHIER_LOG):
        historique = pd.read_csv(FICHIER_LOG)
        historique = pd.concat([historique, ligne], ignore_index=True)
    else:
        historique = ligne
    historique.to_csv(FICHIER_LOG, index=False)


def recuperer_et_preparer_donnees():
    or_data = yf.download("GC=F", period="2y", interval="1d")
    if isinstance(or_data.columns, pd.MultiIndex):
        or_data.columns = or_data.columns.get_level_values(0)

    dollar_data = yf.download("UUP", period="2y", interval="1d")
    if isinstance(dollar_data.columns, pd.MultiIndex):
        dollar_data.columns = dollar_data.columns.get_level_values(0)

    or_data["retour_1"] = or_data["Close"].pct_change(1)
    or_data["retour_5"] = or_data["Close"].pct_change(5)
    or_data["retour_15"] = or_data["Close"].pct_change(15)
    or_data["volatilite"] = or_data["Close"].rolling(window=20).std()
    or_data["moyenne_rapide"] = or_data["Close"].rolling(window=10).mean()
    or_data["moyenne_lente"] = or_data["Close"].rolling(window=50).mean()
    or_data["ecart_moyennes"] = or_data["moyenne_rapide"] - or_data["moyenne_lente"]
    or_data["heure"] = or_data.index.hour
    or_data["volume_relatif"] = or_data["Volume"] / or_data["Volume"].rolling(window=20).mean()

    delta = or_data["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    perte = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / perte
    or_data["rsi"] = 100 - (100 / (1 + rs))

    or_data["prix_dollar"] = dollar_data["Close"].reindex(or_data.index, method="ffill")
    or_data["retour_dollar"] = or_data["prix_dollar"].pct_change(5)

    or_data["futur"] = or_data["Close"].shift(-5)
    or_data["cible"] = (or_data["futur"] > or_data["Close"]).astype(int)

    return or_data.dropna()


def entrainer_modele(donnees):
    taille_train = int(len(donnees) * 0.85)
    train = donnees.iloc[:taille_train]
    modele = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    modele.fit(train[FEATURES], train["cible"])
    return modele


def jours_ecoules(date_str):
    date_entree = datetime.strptime(date_str, "%Y-%m-%d")
    return (datetime.now() - date_entree).days


def main():
    print(f"=== Exécution du {datetime.now().strftime('%Y-%m-%d %H:%M')} ===")

    etat = charger_etat()
    donnees = recuperer_et_preparer_donnees()
    modele = entrainer_modele(donnees)

    aujourdhui = donnees.iloc[-1]
    prix_actuel = aujourdhui["Close"]
    date_str = donnees.index[-1].strftime("%Y-%m-%d")

    if etat["position_ouverte"]:
        prix_entree = etat["prix_entree"]
        variation_prix = (prix_actuel - prix_entree) / prix_entree
        jours = jours_ecoules(etat["date_entree"])
        cout_interet = (LEVIER - 1) * (TAUX_INTERET_ANNUEL / 365) * max(jours, 0)

        if variation_prix >= TAKE_PROFIT:
            variation_capital = (TAKE_PROFIT * LEVIER) - cout_interet
            etat["capital"] *= (1 + variation_capital)
            print(f"TAKE PROFIT touché (levier x{LEVIER}). Nouveau capital : {etat['capital']:.2f} $")
            enregistrer_trade(date_str, "VENTE (take profit)", prix_actuel, etat["capital"])
            etat["position_ouverte"] = False
            etat["prix_entree"] = None
        elif variation_prix <= -STOP_LOSS:
            variation_capital = -(STOP_LOSS * LEVIER) - cout_interet
            etat["capital"] *= max(1 + variation_capital, 0.01)
            print(f"STOP LOSS touché (levier x{LEVIER}). Nouveau capital : {etat['capital']:.2f} $")
            enregistrer_trade(date_str, "VENTE (stop loss)", prix_actuel, etat["capital"])
            etat["position_ouverte"] = False
            etat["prix_entree"] = None
        else:
            print(f"Position toujours ouverte. Variation prix : {variation_prix*100:.2f} % "
                  f"(x{LEVIER} = {variation_prix*LEVIER*100:.2f} % sur le capital)")
    else:
        proba = modele.predict_proba(aujourdhui[FEATURES].values.reshape(1, -1))[0][1]
        print(f"Confiance du modèle (hausse) : {proba*100:.2f} %")

        if proba >= SEUIL_CONFIANCE:
            etat["position_ouverte"] = True
            etat["prix_entree"] = prix_actuel
            etat["date_entree"] = date_str
            print(f"Nouvelle position ouverte à {prix_actuel:.2f} $ (levier x{LEVIER})")
            enregistrer_trade(date_str, "ACHAT", prix_actuel, etat["capital"])
        else:
            print("Pas de signal assez confiant aujourd'hui, on attend.")

    sauvegarder_etat(etat)
    print(f"Capital actuel : {etat['capital']:.2f} $")


if __name__ == "__main__":
    main()
