#!/bin/bash

echo "🚀 Déploiement de l'API de recommandation sur Google Cloud Functions..."

# Assurez-vous d'être dans le bon dossier
cd "$(dirname "$0")"

gcloud functions deploy api-recommandation \
  --gen2 \
  --runtime=python311 \
  --region=us-central1 \
  --source=. \
  --entry-point=recommend \
  --trigger-http \
  --allow-unauthenticated \
  --memory=512Mi

echo "✅ Terminé !"
