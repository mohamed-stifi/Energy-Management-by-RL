#!/bin/bash
# Nettoyer les anciens fichiers de simulation

echo "🧹 Nettoyage des simulations..."

# Supprimer les dossiers Eplus-* à la racine du projet
rm -rf /Eplus-* 2>/dev/null
echo "✅ Suppression des dossiers /Eplus-*"

# Supprimer les anciens dossiers dans notebooks/
rm -rf /workspaces/energy-rl-project/notebooks/Eplus-* 2>/dev/null
echo "✅ Suppression des dossiers notebooks/Eplus-*"

# S'assurer que data/raw existe
mkdir -p /workspaces/energy-rl-project/data/raw
echo "✅ data/raw créé"

# Lister ce qui est dans data/raw
echo ""
echo "📁 Contenu de data/raw/ :"
ls -la /workspaces/energy-rl-project/data/raw/ | head -20

echo ""
echo "✅ Nettoyage terminé !"