import os
import glob
import pandas as pd

# Chemin vers le dossier contenant les CSV
clicks_dir = 'news-portal-user-interactions-by-globocom/clicks'
# Nom du fichier de sortie
output_file = 'click_hour_concatenate_100.csv'

def main():
    # 1. Lister et trier tous les fichiers CSV du dossier
    # On trie pour s'assurer de prendre les 100 premiers de manière déterministe (ex: hour_000 à hour_099)
    search_pattern = os.path.join(clicks_dir, '*.csv')
    all_csv_files = sorted(glob.glob(search_pattern))
    
    # 2. Garder uniquement les 100 premiers
    files_to_concat = all_csv_files[:100]
    
    if not files_to_concat:
        print(f"Aucun fichier CSV trouvé dans le dossier : {clicks_dir}")
        return
        
    print(f"Trouvé {len(files_to_concat)} fichiers à concaténer.")
    
    # 3. Lire chaque fichier et les stocker dans une liste
    df_list = []
    for file in files_to_concat:
        try:
            df = pd.read_csv(file)
            df_list.append(df)
        except Exception as e:
            print(f"Erreur lors de la lecture de {file} : {e}")
            
    # 4. Concaténer et sauvegarder
    if df_list:
        print("Concaténation des DataFrames en cours...")
        concatenated_df = pd.concat(df_list, ignore_index=True)
        
        print(f"Sauvegarde en cours vers : {output_file}...")
        concatenated_df.to_csv(output_file, index=False)
        
        print("\n=== Terminé avec succès ! ===")
        print(f"Nombre de lignes totales : {len(concatenated_df)}")
        print(f"Aperçu des colonnes : {list(concatenated_df.columns)}")
    else:
        print("Aucun DataFrame n'a pu être lu.")

if __name__ == "__main__":
    main()
