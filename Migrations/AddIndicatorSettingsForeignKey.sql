-- Migration: Add foreign key from IndicatorSettings to Controls table
ALTER TABLE IndicatorSettings
ADD ControlId INT NOT NULL;

ALTER TABLE IndicatorSettings
ADD CONSTRAINT FK_IndicatorSettings_Controls
FOREIGN KEY (ControlId) REFERENCES Controls(Id);

CREATE INDEX IDX_IndicatorSettings_ControlId ON IndicatorSettings(ControlId);
