import React from 'react';
import { FormControl, Select, MenuItem, SelectChangeEvent, Box } from '@mui/material';
import { useLanguage } from '../contexts/LanguageContext';

const LanguageSelector: React.FC = () => {
  const { language, setLanguage } = useLanguage();

  const handleChange = (event: SelectChangeEvent) => {
    setLanguage(event.target.value as 'es' | 'fr' | 'en');
  };

  return (
    <Box sx={{ minWidth: 120 }}>
      <FormControl size="small" variant="outlined" sx={{ backgroundColor: 'white', borderRadius: 1 }}>
        <Select
          value={language}
          onChange={handleChange}
          displayEmpty
          sx={{
            '& .MuiSelect-select': {
              padding: '8px 14px',
              display: 'flex',
              alignItems: 'center',
            }
          }}
        >
          <MenuItem value="fr">
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <span>🇫🇷</span> Français
            </Box>
          </MenuItem>
          <MenuItem value="es">
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <span>🇪🇸</span> Español
            </Box>
          </MenuItem>
          <MenuItem value="en">
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <span>🇬🇧</span> English
            </Box>
          </MenuItem>
        </Select>
      </FormControl>
    </Box>
  );
};

export default LanguageSelector;