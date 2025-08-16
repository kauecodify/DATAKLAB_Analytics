# -*- coding: utf-8 -*-
"""

Created on Sat Aug 16 15:40:47 2025

@author: K

"""

import pandas as pd
import numpy as np
import os
import json
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.font import Font
import seaborn as sns
import pyreadstat  # Para SASBDAT
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from PIL import Image, ImageTk  # Adicionado para manipulação de imagens

# -------------------------------------------------------------------------------

class DataKLABApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DataKLAB Analytics")
        self.root.geometry("1280x800")
        self.root.state('zoomed')  # Maximizar janela

        # Configurar paleta de cores genérica
        self.colors = {
            'primary': '#0066cc',   # Azul corporativo
            'secondary': '#999999', # Cinza médio
            'dark': '#333333',      # Cinza escuro
            'light': '#f0f0f0',     # Fundo claro
            'accent': '#ff9900',    # Laranja para destaque
            'white': '#ffffff'      # Branco
        }

        # Configurar fonte 
        self.fonts = {
            'title': ('Arial', 18, 'bold'),
            'header': ('Arial', 14, 'bold'),
            'subheader': ('Arial', 12, 'bold'),
            'body': ('Arial', 10),
            'small': ('Arial', 9)
        }

        self.datasets = {}
        self.current_dataset = None
        self.model = None
        self.scaler = None

        # Carregar e configurar imagem de fundo
        self.bg_image = None
        self.bg_photo = None
        self.load_background_image("op.png")  # Carrega a imagem de fundo

        # Configurar estilo
        self.style = ttk.Style()
        self.configure_styles()

        # Criar interface
        self.create_widgets()

        # Carregar configurações
        self.config_file = "dataklab_config.json"
        self.load_config()

        # Aplicar tema
        self.root.configure(bg=self.colors['light'])
# -------------------------------------------------------------------------------
    def load_background_image(self, image_path):
        """Carrega e configura a imagem de fundo"""
        try:
            # Carrega a imagem usando PIL
            self.bg_image = Image.open(image_path)
            
            # Redimensiona a imagem para o tamanho da tela
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            self.bg_image = self.bg_image.resize((screen_width, screen_height), Image.LANCZOS)
            
            # Converte para formato Tkinter
            self.bg_photo = ImageTk.PhotoImage(self.bg_image)
            
            # Cria um label para a imagem de fundo
            self.background_label = tk.Label(self.root, image=self.bg_photo)
            self.background_label.place(x=0, y=0, relwidth=1, relheight=1)
            self.background_label.lower()  # Coloca no fundo
            
        except Exception as e:
            print(f"Erro ao carregar imagem de fundo: {e}")
            # Se não conseguir carregar a imagem, usa cor sólida
            self.root.configure(bg=self.colors['light'])
# -------------------------------------------------------------------------------
    def configure_styles(self):
        """Configura os estilos"""
        # Configurar tema
        self.style.theme_use('clam')

        # Configurar cores
        self.style.configure('.', background=self.colors['light'], foreground=self.colors['dark'])
        self.style.configure('TFrame', background=self.colors['light'])
        self.style.configure('TLabel', background=self.colors['light'], foreground=self.colors['dark'], font=self.fonts['body'])
        self.style.configure('Header.TLabel', font=self.fonts['header'], foreground=self.colors['primary'])
        self.style.configure('Title.TLabel', font=self.fonts['title'], foreground=self.colors['primary'])

        # Botões
        self.style.configure('TButton',
                            background=self.colors['primary'],
                            foreground=self.colors['white'],
                            font=self.fonts['body'],
                            borderwidth=1,
                            relief='flat',
                            padding=6)
        self.style.map('TButton',
                      background=[('active', '#004d99'), ('pressed', '#003366')],
                      foreground=[('active', self.colors['white']), ('pressed', self.colors['white'])])

        # Comboboxes
        self.style.configure('TCombobox', fieldbackground=self.colors['white'], background=self.colors['white'])

        # Notebook (abas)
        self.style.configure('TNotebook', background=self.colors['light'])
        self.style.configure('TNotebook.Tab',
                            background=self.colors['secondary'],
                            foreground=self.colors['dark'],
                            padding=[10, 5],
                            font=self.fonts['subheader'])
        self.style.map('TNotebook.Tab',
                      background=[('selected', self.colors['primary'])],
                      foreground=[('selected', self.colors['white'])])

        # Treeview (tabela)
        self.style.configure('Treeview',
                            background=self.colors['white'],
                            foreground=self.colors['dark'],
                            fieldbackground=self.colors['white'],
                            rowheight=25,
                            font=self.fonts['small'])
        self.style.configure('Treeview.Heading',
                            background=self.colors['primary'],
                            foreground=self.colors['white'],
                            font=self.fonts['body'],
                            padding=5)
        self.style.map('Treeview', background=[('selected', self.colors['accent'])])
# -------------------------------------------------------------------------------
    def create_widgets(self):
        # Cabeçalho DataKLAB
        header_frame = ttk.Frame(self.root, style='TFrame')
        header_frame.pack(fill=tk.X, padx=10, pady=10)

        # Logo DataKLAB (simulado com texto)
        logo_label = ttk.Label(header_frame, text="DATAKLAB", style='Title.TLabel')
        logo_label.pack(side=tk.LEFT, padx=(0, 20))

        app_title = ttk.Label(header_frame, text="Analytics Platform", style='Header.TLabel')
        app_title.pack(side=tk.LEFT)

        # Painel principal
        main_frame = ttk.Frame(self.root, style='TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Painel esquerdo (controles)
        left_panel = ttk.Frame(main_frame, width=300, style='TFrame')
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # Painel direito (visualização)
        right_panel = ttk.Frame(main_frame, style='TFrame')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # ========== Painel de Controle ==========
        control_frame = ttk.LabelFrame(left_panel, text="Controle de Dados", style='TFrame')
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # Botão para carregar dataset
        ttk.Button(control_frame, text="Carregar Dataset",
                   command=self.load_dataset).pack(fill=tk.X, pady=5)

        # Botão para salvar dataset
        ttk.Button(control_frame, text="Salvar Dataset",
                   command=self.save_dataset).pack(fill=tk.X, pady=5)

        # Seletor de dataset
        ttk.Label(control_frame, text="Dataset Ativo:").pack(anchor=tk.W, pady=(10, 0))
        self.dataset_var = tk.StringVar()
        self.dataset_combobox = ttk.Combobox(control_frame, textvariable=self.dataset_var, state='readonly')
        self.dataset_combobox.pack(fill=tk.X, pady=5)
        self.dataset_combobox.bind('<<ComboboxSelected>>', self.select_dataset)

        # ========== Painel de Análise ==========
        analysis_frame = ttk.LabelFrame(left_panel, text="Análise de Dados", style='TFrame')
        analysis_frame.pack(fill=tk.X, pady=(0, 10))

        # Botões de análise
        ttk.Button(analysis_frame, text="Visualizar Dados",
                   command=self.show_data).pack(fill=tk.X, pady=5)

        ttk.Button(analysis_frame, text="Estatísticas Descritivas",
                   command=self.show_stats).pack(fill=tk.X, pady=5)

        ttk.Button(analysis_frame, text="Informações do Dataset",
                   command=self.show_info).pack(fill=tk.X, pady=5)

        # ========== Painel de Visualização ==========
        viz_frame = ttk.LabelFrame(left_panel, text="Visualização", style='TFrame')
        viz_frame.pack(fill=tk.X, pady=(0, 10))

        # Tipo de gráfico
        ttk.Label(viz_frame, text="Tipo de Gráfico:").pack(anchor=tk.W)
        self.plot_type_var = tk.StringVar(value="histogram")
        plot_types = ["histogram", "boxplot", "scatter", "correlation", "pairplot"]
        self.plot_combobox = ttk.Combobox(viz_frame, textvariable=self.plot_type_var, values=plot_types, state='readonly')
        self.plot_combobox.pack(fill=tk.X, pady=5)

        # Eixo X
        ttk.Label(viz_frame, text="Eixo X:").pack(anchor=tk.W)
        self.x_var = tk.StringVar()
        self.x_combobox = ttk.Combobox(viz_frame, textvariable=self.x_var, state='readonly')
        self.x_combobox.pack(fill=tk.X, pady=5)

        # Eixo Y (para scatter)
        ttk.Label(viz_frame, text="Eixo Y:").pack(anchor=tk.W)
        self.y_var = tk.StringVar()
        self.y_combobox = ttk.Combobox(viz_frame, textvariable=self.y_var, state='readonly')
        self.y_combobox.pack(fill=tk.X, pady=5)

        # Botão para gerar gráfico
        ttk.Button(viz_frame, text="Gerar Gráfico",
                   command=self.generate_plot).pack(fill=tk.X, pady=10)

        # ========== Painel de Machine Learning ==========
        ml_frame = ttk.LabelFrame(left_panel, text="Machine Learning", style='TFrame')
        ml_frame.pack(fill=tk.X)

        # Variável alvo
        ttk.Label(ml_frame, text="Variável Alvo:").pack(anchor=tk.W)
        self.target_var = tk.StringVar()
        self.target_combobox = ttk.Combobox(ml_frame, textvariable=self.target_var, state='readonly')
        self.target_combobox.pack(fill=tk.X, pady=5)

        # Botões ML
        ttk.Button(ml_frame, text="Treinar Perceptron",
                   command=self.train_perceptron).pack(fill=tk.X, pady=5)

        ttk.Button(ml_frame, text="Avaliar Modelo",
                   command=self.evaluate_model).pack(fill=tk.X, pady=5)

        ttk.Button(ml_frame, text="Salvar Modelo",
                   command=self.save_model).pack(fill=tk.X, pady=5)

        # ========== Painel de Visualização de Dados ==========
        # Notebook para múltiplas abas
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Aba para dados
        self.data_frame = ttk.Frame(self.notebook, style='TFrame')
        self.notebook.add(self.data_frame, text="Dados")

        # Treeview para mostrar dados
        self.data_tree = ttk.Treeview(self.data_frame)
        self.data_tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        # Scrollbar
        scrollbar = ttk.Scrollbar(self.data_frame, orient="vertical", command=self.data_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.data_tree.configure(yscrollcommand=scrollbar.set)

        # Aba para gráficos
        self.plot_frame = ttk.Frame(self.notebook, style='TFrame')
        self.notebook.add(self.plot_frame, text="Visualizações")

        # Canvas para matplotlib
        self.figure = plt.figure(figsize=(10, 6), dpi=100, facecolor=self.colors['light'])
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Aba para logs
        self.log_frame = ttk.Frame(self.notebook, style='TFrame')
        self.notebook.add(self.log_frame, text="Logs")

        self.log_text = tk.Text(self.log_frame, wrap="word", bg=self.colors['white'], fg=self.colors['dark'])
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.config(state=tk.DISABLED, font=self.fonts['small'])

        # Status bar
        self.status_var = tk.StringVar(value="Pronto")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W,
                              background=self.colors['primary'], foreground=self.colors['white'],
                              font=self.fonts['small'])
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
# -------------------------------------------------------------------------------
    def log_message(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.status_var.set(message)
# -------------------------------------------------------------------------------
    def load_config(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    self.datasets = json.load(f)
                self.update_dataset_list()
                self.log_message("Configuração carregada com sucesso!")
            except Exception as e:
                self.log_message(f"Erro ao carregar configuração: {str(e)}")
# -------------------------------------------------------------------------------
    def save_config(self):
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.datasets, f, indent=2)
            self.log_message("Configuração salva com sucesso!")
        except Exception as e:
            self.log_message(f"Erro ao salvar configuração: {str(e)}")
# -------------------------------------------------------------------------------
    def update_dataset_list(self):
        datasets = list(self.datasets.keys())
        self.dataset_combobox['values'] = datasets
        if datasets:
            self.dataset_var.set(datasets[0])
            self.select_dataset()
# -------------------------------------------------------------------------------
    def load_dataset(self):
        file_path = filedialog.askopenfilename(
            title="Selecionar arquivo de dados",
            filetypes=[
                ("CSV files", "*.csv"),
                ("Parquet files", "*.parquet"),
                ("Excel files", "*.xlsx;*.xls"),
                ("SAS files", "*.sas7bdat"),
                ("Todos os arquivos", "*.*")
            ]
        )

        if not file_path:
            return

        try:
            # Determinar o tipo de arquivo
            file_name = os.path.basename(file_path)
            dataset_name = os.path.splitext(file_name)[0]

            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            elif file_path.endswith('.sas7bdat'):
                df, meta = pyreadstat.read_sas7bdat(file_path)
            else:
                self.log_message("Formato de arquivo não suportado!")
                return

            # Armazenar dataset
            self.datasets[dataset_name] = {
                'path': file_path,
                'data': df.to_dict(orient='list'),
                'columns': list(df.columns),
                'dtypes': {col: str(df[col].dtype) for col in df.columns}
            }

            self.save_config()
            self.update_dataset_list()
            self.log_message(f"Dataset '{dataset_name}' carregado com sucesso!")

        except Exception as e:
            self.log_message(f"Erro ao carregar dataset: {str(e)}")
# -------------------------------------------------------------------------------
    def save_dataset(self):
        if not self.current_dataset:
            messagebox.showerror("Erro", "Nenhum dataset selecionado!")
            return

        file_path = filedialog.asksaveasfilename(
            title="Salvar dataset",
            filetypes=[
                ("Parquet files", "*.parquet"),
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx"),
                ("SAS files", "*.sas7bdat"),
                ("Power BI files", "*.pbix")
            ]
        )

        if not file_path:
            return

        try:
            df = self.get_current_df()

            if file_path.endswith('.parquet'):
                df.to_parquet(file_path)
                self.log_message(f"Dataset salvo em formato Parquet: {file_path}")

            elif file_path.endswith('.csv'):
                df.to_csv(file_path, index=False)
                self.log_message(f"Dataset salvo em formato CSV: {file_path}")

            elif file_path.endswith(('.xlsx', '.xls')):
                df.to_excel(file_path, index=False)
                self.log_message(f"Dataset salvo em formato Excel: {file_path}")

            elif file_path.endswith('.sas7bdat'):
                # Converter tipos de dados para compatibilidade com SAS
                sas_df = df.copy()
                for col in sas_df.columns:
                    if sas_df[col].dtype == 'object':
                        sas_df[col] = sas_df[col].astype(str)
                    elif pd.api.types.is_datetime64_any_dtype(sas_df[col]):
                        sas_df[col] = sas_df[col].dt.strftime('%Y-%m-%d')

                pyreadstat.write_sas7bdat(sas_df, file_path)
                self.log_message(f"Dataset salvo em formato SAS: {file_path}")

            elif file_path.endswith('.pbix'):
                # Power BI não permite salvar dados diretamente em PBIX
                # Como alternativa, salvamos em formato suportado pelo Power BI
                base_path = os.path.splitext(file_path)[0]
                df.to_parquet(f"{base_path}.parquet")
                self.log_message((
                    "Power BI não permite salvar dados diretamente em PBIX.\n"
                    f"Os dados foram salvos em formato Parquet: {base_path}.parquet\n"
                    "Você pode importar este arquivo no Power BI."
                ))

            else:
                self.log_message("Formato não suportado!")

        except Exception as e:
            self.log_message(f"Erro ao salvar dataset: {str(e)}")
# -------------------------------------------------------------------------------
    def select_dataset(self, event=None):
        dataset_name = self.dataset_var.get()
        if not dataset_name or dataset_name not in self.datasets:
            return

        self.current_dataset = dataset_name
        data_info = self.datasets[dataset_name]

        # Atualizar comboboxes
        self.x_combobox['values'] = data_info['columns']
        self.y_combobox['values'] = data_info['columns']
        self.target_combobox['values'] = data_info['columns']

        # Selecionar primeiros valores
        if data_info['columns']:
            self.x_var.set(data_info['columns'][0])
            self.target_var.set(data_info['columns'][-1])

            if len(data_info['columns']) > 1:
                self.y_var.set(data_info['columns'][1])

        self.log_message(f"Dataset '{dataset_name}' selecionado")
# -------------------------------------------------------------------------------
    def get_current_df(self):
        if not self.current_dataset:
            return None

        data_info = self.datasets[self.current_dataset]
        df = pd.DataFrame(data_info['data'])

        # Converter tipos de dados
        for col, dtype in data_info['dtypes'].items():
            if dtype == 'category':
                df[col] = df[col].astype('category')
            elif 'int' in dtype:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')
            elif 'float' in dtype:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float')
            elif 'datetime' in dtype:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        return df
# -------------------------------------------------------------------------------
    def show_data(self):
        if not self.current_dataset:
            messagebox.showerror("Erro", "Nenhum dataset selecionado!")
            return

        df = self.get_current_df()
        if df is None:
            return

        # Limpar treeview
        self.data_tree.delete(*self.data_tree.get_children())

        # Configurar colunas
        columns = list(df.columns)
        self.data_tree['columns'] = columns
        self.data_tree['show'] = 'headings'

        for col in columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=100, anchor=tk.CENTER)

        # Adicionar dados (amostra de 100 linhas)
        sample = df.head(100)
        for _, row in sample.iterrows():
            self.data_tree.insert('', tk.END, values=list(row))

        self.notebook.select(self.data_frame)
        self.log_message("Dados exibidos")
# -------------------------------------------------------------------------------
    def show_stats(self):
        if not self.current_dataset:
            messagebox.showerror("Erro", "Nenhum dataset selecionado!")
            return

        df = self.get_current_df()
        if df is None:
            return

        stats = df.describe(include='all').T
        stats['dtype'] = df.dtypes
        stats['null'] = df.isnull().sum()
        stats['unique'] = df.nunique()

        # Criar nova janela para estatísticas
        stats_window = tk.Toplevel(self.root)
        stats_window.title("Estatísticas Descritivas")
        stats_window.geometry("800x600")
        stats_window.configure(bg=self.colors['light'])

        # Treeview para estatísticas
        tree = ttk.Treeview(stats_window, style='Treeview')
        tree.pack(fill=tk.BOTH, expand=True)

        # Configurar colunas
        stat_cols = list(stats.columns)
        tree['columns'] = stat_cols
        tree['show'] = 'headings'

        for col in stat_cols:
            tree.heading(col, text=col)
            tree.column(col, width=100, anchor=tk.CENTER)

        # Adicionar dados
        for idx, row in stats.iterrows():
            tree.insert('', tk.END, text=idx, values=list(row))

        self.log_message("Estatísticas descritivas geradas")
# -------------------------------------------------------------------------------
    def show_info(self):
        if not self.current_dataset:
            messagebox.showerror("Erro", "Nenhum dataset selecionado!")
            return

        data_info = self.datasets[self.current_dataset]

        # Criar nova janela para informações
        info_window = tk.Toplevel(self.root)
        info_window.title("Informações do Dataset")
        info_window.geometry("600x400")
        info_window.configure(bg=self.colors['light'])

        # Frame para informações
        info_frame = ttk.Frame(info_window, style='TFrame')
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Labels com informações
        ttk.Label(info_frame, text=f"Dataset: {self.current_dataset}", style='Header.TLabel').pack(anchor=tk.W, pady=(0, 10))
        ttk.Label(info_frame, text=f"Arquivo: {data_info['path']}").pack(anchor=tk.W)
        ttk.Label(info_frame, text=f"Número de colunas: {len(data_info['columns'])}").pack(anchor=tk.W)

        # Tipos de dados
        ttk.Label(info_frame, text="Tipos de Dados:", style='Header.TLabel').pack(anchor=tk.W, pady=(10, 0))
        for col, dtype in data_info['dtypes'].items():
            ttk.Label(info_frame, text=f"{col}: {dtype}").pack(anchor=tk.W)

        self.log_message("Informações do dataset exibidas")
# -------------------------------------------------------------------------------
    def generate_plot(self):
        if not self.current_dataset:
            messagebox.showerror("Erro", "Nenhum dataset selecionado!")
            return

        df = self.get_current_df()
        if df is None:
            return

        plot_type = self.plot_type_var.get()
        x_col = self.x_var.get()

        if not x_col:
            messagebox.showerror("Erro", "Selecione uma coluna para o eixo X!")
            return

        try:
            self.figure.clear()
            ax = self.figure.add_subplot(111)

            # Configurar cores do gráfico
            dataklab_colors = [self.colors['primary'], self.colors['dark'], self.colors['accent']]
            plt.rcParams['axes.prop_cycle'] = plt.cycler(color=dataklab_colors)

            if plot_type == "histogram":
                df[x_col].hist(ax=ax, bins=20)
                ax.set_title(f"Histograma de {x_col}", fontsize=14, color=self.colors['primary'])
                ax.set_xlabel(x_col)
                ax.set_ylabel("Frequência")
                ax.grid(True, linestyle='--', alpha=0.7)

            elif plot_type == "boxplot":
                df.boxplot(column=x_col, ax=ax)
                ax.set_title(f"Boxplot de {x_col}", fontsize=14, color=self.colors['primary'])
                ax.grid(True, linestyle='--', alpha=0.7)

            elif plot_type == "scatter":
                y_col = self.y_var.get()
                if not y_col:
                    messagebox.showerror("Erro", "Selecione uma coluna para o eixo Y!")
                    return

                df.plot.scatter(x=x_col, y=y_col, ax=ax, alpha=0.7)
                ax.set_title(f"Scatter Plot: {x_col} vs {y_col}", fontsize=14, color=self.colors['primary'])
                ax.grid(True, linestyle='--', alpha=0.7)

            elif plot_type == "correlation":
                corr = df.corr(numeric_only=True)
                sns.heatmap(corr, annot=True, fmt=".2f", ax=ax, cmap="coolwarm")
                ax.set_title("Matriz de Correlação", fontsize=14, color=self.colors['primary'])

            elif plot_type == "pairplot":
                # Amostrar para não sobrecarregar
                sample = df.sample(min(100, len(df)))
                sns.pairplot(sample, palette=dataklab_colors)
                self.canvas.draw()
                return

            # Configurar aparência geral
            for spine in ax.spines.values():
                spine.set_edgecolor(self.colors['secondary'])
            ax.tick_params(colors=self.colors['dark'])
            ax.set_facecolor(self.colors['light'])
            self.figure.set_facecolor(self.colors['light'])

            self.canvas.draw()
            self.notebook.select(self.plot_frame)
            self.log_message(f"Gráfico {plot_type} gerado para {x_col}")

        except Exception as e:
            self.log_message(f"Erro ao gerar gráfico: {str(e)}")
# -------------------------------------------------------------------------------
    def train_perceptron(self):
        if not self.current_dataset:
            messagebox.showerror("Erro", "Nenhum dataset selecionado!")
            return

        target_col = self.target_var.get()
        if not target_col:
            messagebox.showerror("Erro", "Selecione uma coluna alvo!")
            return

        df = self.get_current_df()
        if df is None:
            return

        try:
            # Preparar dados
            X = df.drop(columns=[target_col])
            y = df[target_col]

            # Converter variáveis categóricas
            X = pd.get_dummies(X)

            # Codificar target se necessário
            if y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y)

            # Dividir dados
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )

            # Escalonar
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

            # Treinar modelo
            self.model = Perceptron(
                max_iter=1000,
                eta0=0.1,
                random_state=42,
                early_stopping=True
            )

            self.model.fit(X_train, y_train)

            # Avaliar
            train_acc = self.model.score(X_train, y_train)
            test_acc = self.model.score(X_test, y_test)

            self.log_message(f"Perceptron treinado com sucesso!")
            self.log_message(f"Acurácia treino: {train_acc:.4f}")
            self.log_message(f"Acurácia teste: {test_acc:.4f}")

            # Mostrar mensagem de sucesso
            messagebox.showinfo("Treino Concluído",
                              f"Modelo treinado com sucesso!\nAcurácia treino: {train_acc:.4f}\nAcurácia teste: {test_acc:.4f}")

        except Exception as e:
            self.log_message(f"Erro ao treinar Perceptron: {str(e)}")
            messagebox.showerror("Erro", f"Falha no treinamento: {str(e)}")
# -------------------------------------------------------------------------------
    def evaluate_model(self):
        if not self.model:
            messagebox.showerror("Erro", "Nenhum modelo treinado!")
            return

        if not self.current_dataset:
            messagebox.showerror("Erro", "Nenhum dataset selecionado!")
            return

        target_col = self.target_var.get()
        if not target_col:
            messagebox.showerror("Erro", "Selecione uma coluna alvo!")
            return

        df = self.get_current_df()
        if df is None:
            return

        try:
            # Preparar dados
            X = df.drop(columns=[target_col])
            y = df[target_col]

            # Converter variáveis categóricas
            X = pd.get_dummies(X)

            # Codificar target se necessário
            if y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y)

            # Escalonar
            if self.scaler:
                X_scaled = self.scaler.transform(X)
            else:
                self.log_message("Nenhum scaler disponível, usando dados brutos")
                X_scaled = X.values

            # Avaliar
            accuracy = self.model.score(X_scaled, y)

            self.log_message(f"Acurácia do modelo: {accuracy:.4f}")

            # Mostrar resultado em uma messagebox
            messagebox.showinfo("Avaliação do Modelo",
                              f"Acurácia geral do modelo: {accuracy:.4f}")

        except Exception as e:
            self.log_message(f"Erro ao avaliar modelo: {str(e)}")
            messagebox.showerror("Erro", f"Falha na avaliação: {str(e)}")
# -------------------------------------------------------------------------------
    def save_model(self):
        if not self.model:
            messagebox.showerror("Erro", "Nenhum modelo treinado!")
            return

        file_path = filedialog.asksaveasfilename(
            title="Salvar modelo",
            defaultextension=".joblib",
            filetypes=[
                ("Joblib files", "*.joblib"),
                ("Pickle files", "*.pkl"),
                ("Todos os arquivos", "*.*")
            ]
        )

        if not file_path:
            return

        try:
            if file_path.endswith('.joblib'):
                joblib.dump({
                    'model': self.model,
                    'scaler': self.scaler
                }, file_path)
            else:
                import pickle
                with open(file_path, 'wb') as f:
                    pickle.dump({
                        'model': self.model,
                        'scaler': self.scaler
                    }, f)

            self.log_message(f"Modelo salvo em: {file_path}")
            messagebox.showinfo("Sucesso", f"Modelo salvo com sucesso em:\n{file_path}")
        except Exception as e:
            self.log_message(f"Erro ao salvar modelo: {str(e)}")
            messagebox.showerror("Erro", f"Falha ao salvar modelo: {str(e)}")
# -------------------------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = DataKLABApp(root)
    root.mainloop()
# -------------------------------------------------------------------------------