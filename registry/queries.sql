-- 1. Candidats pouvant potentiellement accepter les 32 canaux ASC.
SELECT *
FROM asc_model_catalog
WHERE "Peut recevoir mes 32 canaux ASC" IN ('oui', 'à vérifier')
ORDER BY "Score de compatibilité" DESC;

-- 2. Candidats ayant des poids, mais nécessitant une adaptation spatiale.
SELECT
    "Nom du modèle",
    "Nombre de canaux du pré-entraînement",
    "Noms/positions des électrodes utilisés",
    "Justification",
    "URL"
FROM asc_model_catalog
WHERE "Poids réellement disponibles" = 'oui'
  AND "Peut recevoir mes 32 canaux ASC" = 'à adapter'
ORDER BY "Score de compatibilité" DESC;

-- 3. Vérifier la provenance de chaque information d'un candidat.
SELECT
    c.display_name,
    e.field_name,
    e.value_text,
    e.source_url,
    e.extraction_method,
    e.confidence
FROM evidence e
JOIN candidate c ON c.candidate_id = e.candidate_id
WHERE c.display_name LIKE '%SignalJEPA%'
ORDER BY e.field_name;

-- 4. Voir les fichiers qui prouvent l'existence de poids.
SELECT
    c.display_name,
    w.file_path,
    w.file_format,
    w.size_bytes,
    w.download_url
FROM weight_file w
JOIN candidate c ON c.candidate_id = w.candidate_id
ORDER BY c.display_name, w.file_path;

