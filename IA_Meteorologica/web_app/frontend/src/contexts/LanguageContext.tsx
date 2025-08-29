import React, { createContext, useState, useContext, ReactNode } from 'react';

interface Translations {
  // Navigation
  systemTitle: string;
  myProfile: string;
  logout: string;
  login: string;
  register: string;
  
  // Model Selection
  modelSelection: string;
  selectModel: string;
  lstm: string;
  lstmDesc: string;
  cnn: string;
  cnnDesc: string;
  decisionTree: string;
  decisionTreeDesc: string;
  transformer: string;
  transformerDesc: string;
  randomForest: string;
  randomForestDesc: string;
  nbeats: string;
  nbeatsDesc: string;
  nhits: string;
  nhitsDesc: string;
  
  // Common Actions
  next: string;
  back: string;
  save: string;
  cancel: string;
  load: string;
  upload: string;
  download: string;
  train: string;
  predict: string;
  
  // Data & Variables
  dataset: string;
  variables: string;
  normalization: string;
  hyperparameters: string;
  metrics: string;
  trainTestSplit: string;
  
  // Authentication
  username: string;
  password: string;
  confirmPassword: string;
  email: string;
  createAccount: string;
  welcomeMessage: string;
  loginPrompt: string;
  registerPrompt: string;
  noAccount: string;
  hasAccount: string;
  
  // Messages
  loginError: string;
  logoutSuccess: string;
  passwordLength: string;
  passwordMatch: string;
  registrationSuccess: string;
}

const translations: Record<'es' | 'fr' | 'en', Translations> = {
  es: {
    // Navigation
    systemTitle: 'Sistema de Predicción Meteorológica con IA',
    myProfile: 'Mi Perfil',
    logout: 'Cerrar Sesión',
    login: 'Iniciar Sesión',
    register: 'Registrar',
    
    // Model Selection
    modelSelection: 'Selección de Modelo',
    selectModel: 'Selecciona un Modelo',
    lstm: 'LSTM',
    lstmDesc: 'Ideal para series temporales y predicciones meteorológicas',
    cnn: 'CNN',
    cnnDesc: 'Para patrones espaciales en datos meteorológicos',
    decisionTree: 'Árbol de Decisiones',
    decisionTreeDesc: 'Modelo interpretable para relaciones no lineales simples',
    transformer: 'Transformer',
    transformerDesc: 'Arquitectura de atención para capturar dependencias complejas',
    randomForest: 'Random Forest',
    randomForestDesc: 'Conjunto de árboles para predicciones robustas',
    nbeats: 'N-BEATS',
    nbeatsDesc: 'Neural Basis Expansion Analysis for Time Series',
    nhits: 'N-HiTS',
    nhitsDesc: 'Neural Hierarchical Interpolation for Time Series',
    
    // Common Actions
    next: 'Siguiente',
    back: 'Atrás',
    save: 'Guardar',
    cancel: 'Cancelar',
    load: 'Cargar',
    upload: 'Subir',
    download: 'Descargar',
    train: 'Entrenar',
    predict: 'Predecir',
    
    // Data & Variables
    dataset: 'Conjunto de Datos',
    variables: 'Variables',
    normalization: 'Normalización',
    hyperparameters: 'Hiperparámetros',
    metrics: 'Métricas',
    trainTestSplit: 'División Entrenamiento/Prueba',
    
    // Authentication
    username: 'Usuario',
    password: 'Contraseña',
    confirmPassword: 'Confirmar contraseña',
    email: 'Correo electrónico',
    createAccount: 'Crear Cuenta',
    welcomeMessage: 'Únete a Weather AI para empezar a predecir el clima',
    loginPrompt: 'Inicia sesión para acceder a tu cuenta',
    registerPrompt: 'Crea tu cuenta para comenzar',
    noAccount: '¿No tienes cuenta? Regístrate aquí',
    hasAccount: '¿Ya tienes cuenta? Inicia sesión aquí',
    
    // Messages
    loginError: 'Usuario o contraseña incorrectos.',
    logoutSuccess: 'Has cerrado sesión exitosamente.',
    passwordLength: 'Tu contraseña debe tener al menos 8 caracteres.',
    passwordMatch: 'Las contraseñas no coinciden.',
    registrationSuccess: '¡Bienvenido {username}! Tu cuenta ha sido creada exitosamente.',
  },
  fr: {
    // Navigation
    systemTitle: 'Système de Prédiction Météorologique avec IA',
    myProfile: 'Mon Profil',
    logout: 'Déconnexion',
    login: 'Connexion',
    register: "S'inscrire",
    
    // Model Selection
    modelSelection: 'Sélection du Modèle',
    selectModel: 'Sélectionnez un Modèle',
    lstm: 'LSTM',
    lstmDesc: 'Idéal pour les séries temporelles et les prévisions météorologiques',
    cnn: 'CNN',
    cnnDesc: 'Pour les modèles spatiaux dans les données météorologiques',
    decisionTree: 'Arbre de Décision',
    decisionTreeDesc: 'Modèle interprétable pour les relations non linéaires simples',
    transformer: 'Transformer',
    transformerDesc: "Architecture d'attention pour capturer les dépendances complexes",
    randomForest: 'Forêt Aléatoire',
    randomForestDesc: "Ensemble d'arbres pour des prédictions robustes",
    nbeats: 'N-BEATS',
    nbeatsDesc: 'Analyse d\'Expansion de Base Neuronale pour les Séries Temporelles',
    nhits: 'N-HiTS',
    nhitsDesc: 'Interpolation Hiérarchique Neuronale pour les Séries Temporelles',
    
    // Common Actions
    next: 'Suivant',
    back: 'Retour',
    save: 'Enregistrer',
    cancel: 'Annuler',
    load: 'Charger',
    upload: 'Télécharger',
    download: 'Télécharger',
    train: 'Entraîner',
    predict: 'Prédire',
    
    // Data & Variables
    dataset: 'Ensemble de Données',
    variables: 'Variables',
    normalization: 'Normalisation',
    hyperparameters: 'Hyperparamètres',
    metrics: 'Métriques',
    trainTestSplit: 'Division Entraînement/Test',
    
    // Authentication
    username: 'Utilisateur',
    password: 'Mot de passe',
    confirmPassword: 'Confirmer le mot de passe',
    email: 'Adresse e-mail',
    createAccount: 'Créer un compte',
    welcomeMessage: 'Rejoignez Weather AI pour commencer à prédire la météo',
    loginPrompt: 'Connectez-vous pour accéder à votre compte',
    registerPrompt: 'Créez votre compte pour commencer',
    noAccount: "Vous n'avez pas de compte ? Inscrivez-vous ici",
    hasAccount: 'Vous avez déjà un compte ? Connectez-vous ici',
    
    // Messages
    loginError: 'Utilisateur ou mot de passe incorrect.',
    logoutSuccess: 'Vous avez été déconnecté avec succès.',
    passwordLength: 'Votre mot de passe doit contenir au moins 8 caractères.',
    passwordMatch: 'Les mots de passe ne correspondent pas.',
    registrationSuccess: 'Bienvenue {username} ! Votre compte a été créé avec succès.',
  },
  en: {
    // Navigation
    systemTitle: 'Weather Prediction System with AI',
    myProfile: 'My Profile',
    logout: 'Logout',
    login: 'Login',
    register: 'Register',
    
    // Model Selection
    modelSelection: 'Model Selection',
    selectModel: 'Select a Model',
    lstm: 'LSTM',
    lstmDesc: 'Ideal for time series and weather predictions',
    cnn: 'CNN',
    cnnDesc: 'For spatial patterns in meteorological data',
    decisionTree: 'Decision Tree',
    decisionTreeDesc: 'Interpretable model for simple nonlinear relationships',
    transformer: 'Transformer',
    transformerDesc: 'Attention architecture to capture complex dependencies',
    randomForest: 'Random Forest',
    randomForestDesc: 'Ensemble of trees for robust predictions',
    nbeats: 'N-BEATS',
    nbeatsDesc: 'Neural Basis Expansion Analysis for Time Series',
    nhits: 'N-HiTS',
    nhitsDesc: 'Neural Hierarchical Interpolation for Time Series',
    
    // Common Actions
    next: 'Next',
    back: 'Back',
    save: 'Save',
    cancel: 'Cancel',
    load: 'Load',
    upload: 'Upload',
    download: 'Download',
    train: 'Train',
    predict: 'Predict',
    
    // Data & Variables
    dataset: 'Dataset',
    variables: 'Variables',
    normalization: 'Normalization',
    hyperparameters: 'Hyperparameters',
    metrics: 'Metrics',
    trainTestSplit: 'Train/Test Split',
    
    // Authentication
    username: 'Username',
    password: 'Password',
    confirmPassword: 'Confirm Password',
    email: 'Email',
    createAccount: 'Create Account',
    welcomeMessage: 'Join Weather AI to start predicting weather',
    loginPrompt: 'Sign in to access your account',
    registerPrompt: 'Create your account to get started',
    noAccount: "Don't have an account? Register here",
    hasAccount: 'Already have an account? Login here',
    
    // Messages
    loginError: 'Incorrect username or password.',
    logoutSuccess: 'You have successfully logged out.',
    passwordLength: 'Your password must be at least 8 characters.',
    passwordMatch: 'Passwords do not match.',
    registrationSuccess: 'Welcome {username}! Your account has been created successfully.',
  },
};

interface LanguageContextType {
  language: 'es' | 'fr' | 'en';
  setLanguage: (lang: 'es' | 'fr' | 'en') => void;
  t: Translations;
}

const LanguageContext = createContext<LanguageContextType | undefined>(undefined);

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [language, setLanguage] = useState<'es' | 'fr' | 'en'>('fr'); // French as default

  const value = {
    language,
    setLanguage,
    t: translations[language],
  };

  return (
    <LanguageContext.Provider value={value}>
      {children}
    </LanguageContext.Provider>
  );
}

export function useLanguage() {
  const context = useContext(LanguageContext);
  if (context === undefined) {
    throw new Error('useLanguage must be used within a LanguageProvider');
  }
  return context;
}