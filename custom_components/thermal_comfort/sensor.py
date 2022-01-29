"""Calculates comfort values from temperature and humidity."""
from __future__ import annotations

import logging
import math
from typing import Any, Final

from homeassistant import util
from homeassistant.components.sensor import PLATFORM_SCHEMA, SensorEntity
from homeassistant.const import (
    ATTR_TEMPERATURE,
    ATTR_UNIT_OF_MEASUREMENT,
    CONF_SENSORS,
    DEVICE_CLASS_HUMIDITY,
    DEVICE_CLASS_TEMPERATURE,
    EVENT_HOMEASSISTANT_START,
    PERCENTAGE,
    STATE_UNKNOWN,
    TEMP_CELSIUS,
    TEMP_FAHRENHEIT,
)
from homeassistant.core import Event, HomeAssistant, State, callback
import homeassistant.helpers.config_validation as cv
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
import voluptuous as vol

_LOGGER = logging.getLogger(__name__)

ATTR_HUMIDITY: Final = "humidity"

GRAMS_PER_CUBIC_METER: Final = "g/m³"

CONF_TEMPERATURE_SENSOR: Final = "temperature_sensor"
CONF_HUMIDITY_SENSOR: Final = "humidity_sensor"
CONF_SENSOR_TYPES: Final = "sensor_types"
CONF_FRIENDLY_NAME: Final = "friendly_name"
CONF_CHANGE_THRESHOLD: Final = "change_threshold"

SENSOR_ABSOLUTE_HUMIDITY: Final = "absolutehumidity"
SENSOR_HEAT_INDEX: Final = "heatindex"
SENSOR_DEW_POINT: Final = "dewpoint"
SENSOR_PERCEPTION: Final = "perception"
SENSOR_SIMMER_INDEX: Final = "simmerindex"
SENSOR_SIMMER_DANGER: Final = "simmerdanger"

SENSOR_TYPES: Final = {
    SENSOR_ABSOLUTE_HUMIDITY: ("Absolute Humidity", DEVICE_CLASS_HUMIDITY),
    SENSOR_HEAT_INDEX: ("Heat Index", DEVICE_CLASS_TEMPERATURE),
    SENSOR_DEW_POINT: ("Dewpoint", DEVICE_CLASS_TEMPERATURE),
    SENSOR_PERCEPTION: ("Perception", None),
    SENSOR_SIMMER_INDEX: ("SSI", DEVICE_CLASS_TEMPERATURE),
    SENSOR_SIMMER_DANGER: ("SSI Danger", None),
}

SENSOR_SCHEMA: Final = vol.Schema(
    {
        vol.Required(CONF_TEMPERATURE_SENSOR): cv.entity_id,
        vol.Required(CONF_HUMIDITY_SENSOR): cv.entity_id,
        vol.Optional(CONF_SENSOR_TYPES, default=SENSOR_TYPES.keys()): cv.ensure_list,
        vol.Optional(CONF_FRIENDLY_NAME): cv.string,
        vol.Optional(CONF_CHANGE_THRESHOLD, default=0.1): cv.positive_float,
    }
)

PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend(
    {
        vol.Required(CONF_SENSORS): cv.schema_with_slug_keys(SENSOR_SCHEMA),
    }
)


async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None,
) -> None:
    """Set up comfort sensor."""

    sensors = []

    for name, device_config in config[CONF_SENSORS].items():
        temp_sensor = device_config.get(CONF_TEMPERATURE_SENSOR)
        humidity_sensor = device_config.get(CONF_HUMIDITY_SENSOR)
        config_sensor_types = device_config.get(CONF_SENSOR_TYPES)
        friendly_name = device_config.get(CONF_FRIENDLY_NAME, name)
        change_threshold = device_config.get(CONF_CHANGE_THRESHOLD)

        for sensor_type in config_sensor_types[0]:
            if sensor_type not in SENSOR_TYPES:
                continue
            description, device_class = SENSOR_TYPES[sensor_type]
            sensors.append(
                ComfortSensor(
                    hass,
                    f"{friendly_name} {description}",
                    device_class,
                    sensor_type,
                    temp_sensor,
                    humidity_sensor,
                    change_threshold,
                )
            )

    if not sensors:
        _LOGGER.error("No sensors added")
        return False

    async_add_entities(sensors, False)


class ComfortSensor(SensorEntity):
    """Represents a comfort sensor."""

    def __init__(
        self,
        hass: HomeAssistant,
        name: str,
        device_class: str,
        sensor_type: str,
        temp_sensor: Entity,
        humidity_sensor: Entity,
        change_threshold: float,
    ) -> None:
        """Initialize the sensor."""
        self._name: Final = name
        self._device_class: Final = device_class
        self._sensor_type: Final = sensor_type
        self._temp_sensor: Final = temp_sensor
        self._humidity_sensor: Final = humidity_sensor
        self._change_threshold: Final = change_threshold

        if self._device_class == DEVICE_CLASS_TEMPERATURE:
            self._unit_of_measurement = hass.config.units.temperature_unit
        elif self._sensor_type == SENSOR_ABSOLUTE_HUMIDITY:
            self._unit_of_measurement = GRAMS_PER_CUBIC_METER
        else:
            self._unit_of_measurement = None

        self._available: bool = False
        self._state: float = None
        self._temp_c: float = None
        self._humidity: float = None

    async def async_added_to_hass(self) -> None:
        """Register callbacks."""

        @callback
        def comfort_sensors_state_listener(event: Event) -> None:
            """Handle for state changes for dependent entities."""
            new_state = event.data.get("new_state")
            old_state = event.data.get("old_state")
            entity = event.data.get("entity_id")
            _LOGGER.debug(
                f"Sensor state change for {entity} that had old state {old_state} and new state {new_state}",
            )

            if (
                self._update_sensor(entity, old_state, new_state)
                and self._calculate_state()
            ):
                self.async_schedule_update_ha_state(True)

        @callback
        def comfort_startup(event: Event) -> None:
            """Add listeners and get 1st state."""
            _LOGGER.debug(f"Startup for {self.entity_id}")

            async_track_state_change_event(
                self.hass,
                (self._temp_sensor, self._humidity_sensor),
                comfort_sensors_state_listener,
            )

            # Read initial state
            schedule_update = True

            temperature = self.hass.states.get(self._temp_sensor)
            schedule_update &= self._update_sensor(self._temp_sensor, None, temperature)

            humidity = self.hass.states.get(self._humidity_sensor)
            schedule_update &= self._update_sensor(
                self._humidity_sensor, None, humidity
            )

            schedule_update &= self._calculate_state()

            if schedule_update:
                self.async_schedule_update_ha_state(True)

        self.hass.bus.async_listen_once(EVENT_HOMEASSISTANT_START, comfort_startup)

    def _update_sensor(
        self, sensor: Entity, old_state: State, new_state: State
    ) -> bool:
        """Update information based on new sensor states."""
        _LOGGER.debug(f"Sensor update for {sensor}")
        if new_state is None:
            return False

        # If old_state is not set and new state is unknown then it means
        # that the sensor just started up
        if old_state is None and new_state.state == STATE_UNKNOWN:
            return False

        if sensor == self._temp_sensor:
            self._update_temperature(new_state)
        elif sensor == self._humidity_sensor:
            self._update_humidity(new_state)

        return True

    def _update_temperature(self, state: State) -> None:
        """Parse temperature sensor value."""
        _LOGGER.debug(f"Updating temp sensor with value {state.state}")

        # Return an error if the sensor change its state to Unknown.
        if state.state == STATE_UNKNOWN:
            _LOGGER.debug(
                f"Unable to parse temperature sensor {state.entity_id} with state: {state.state}",
            )
            return

        unit = state.attributes.get(ATTR_UNIT_OF_MEASUREMENT)
        temp = util.convert(state.state, float)

        if temp is None:
            _LOGGER.debug(
                f"Unable to parse temperature sensor {state.entity_id} with state: {state.state}",
            )
            return

        if unit == TEMP_FAHRENHEIT:
            self._temp_c = util.temperature.fahrenheit_to_celsius(temp)
        elif unit == TEMP_CELSIUS:
            self._temp_c = temp
        else:
            _LOGGER.error(
                f"Temp sensor {state.entity_id} has unsupported unit: {unit} (allowed: {TEMP_CELSIUS}, {TEMP_FAHRENHEIT})",
            )

    def _update_humidity(self, state: State) -> None:
        """Parse humidity sensor value."""
        _LOGGER.debug(f"Updating humidity sensor with value {state.state}")

        # Return an error if the sensor change its state to Unknown.
        if state.state == STATE_UNKNOWN:
            _LOGGER.debug(
                f"Unable to parse humidity sensor {state.entity_id}, state: {state.state}",
            )
            return

        unit = state.attributes.get(ATTR_UNIT_OF_MEASUREMENT)
        humidity = util.convert(state.state, float)

        if humidity is None:
            _LOGGER.debug(
                f"Unable to parse humidity sensor {state.entity_id}, state: {state.state}",
            )
            return

        if unit != PERCENTAGE:
            _LOGGER.error(
                f"Humidity sensor {state.entity_id} has unsupported unit: {unit} (allowed: {PERCENTAGE})",
            )
            return

        if humidity > 100 or humidity < 0:
            _LOGGER.error(
                f"Humidity sensor {state.entity_id} is out of range: {humidity} (allowed: 0-100%)",
            )
            return

        self._humidity = humidity

    def _calculate_state(self) -> bool:
        if None in (self._temp_c, self._humidity):
            return False

        if self._sensor_type == SENSOR_DEW_POINT:
            new_state = self._calc_dew_point()
        elif self._sensor_type == SENSOR_HEAT_INDEX:
            new_state = self._calc_heat_index()
        elif self._sensor_type == SENSOR_ABSOLUTE_HUMIDITY:
            new_state = self._calc_absolute_humidity()
        elif self._sensor_type == SENSOR_PERCEPTION:
            new_state = self._calc_thermal_perception()
        elif self._sensor_type == SENSOR_SIMMER_INDEX:
            new_state = self._calc_simmer_index()
        elif self._sensor_type == SENSOR_SIMMER_DANGER:
            new_state = self._calc_simmer_danger()

        if new_state == self._state:
            return False

        if (
            isinstance(new_state, float)
            and isinstance(self._state, float)
            and abs(new_state - self._state) < self._change_threshold
        ):
            _LOGGER.debug(
                f"Suppressed {self._name} update less than {self._change_threshold} - new: {new_state} current: {self._state}",
            )
            return False

        self._state = new_state
        self._available = new_state is not None
        return True

    def _calc_dew_point(self) -> float:
        """Calculate the dew point."""

        temp_c = self._temp_c
        humidity = self._humidity

        # https://en.wikipedia.org/wiki/Arden_Buck_equation
        if temp_c < 0:
            saturation_vapor_pressure = 6.1115 * math.exp(
                (23.036 - (temp_c / 333.7)) * (temp_c / (279.82 + temp_c))
            )
        else:
            saturation_vapor_pressure = 6.1121 * math.exp(
                (18.678 - (temp_c / 234.5)) * (temp_c / (257.14 + temp_c))
            )
        vapor_pressure = saturation_vapor_pressure * (humidity / 100.0)
        dew_point_c = (-430.22 + 237.7 * math.log(vapor_pressure)) / (
            -math.log(vapor_pressure) + 19.08
        )
        return dew_point_c

    def _calc_thermal_perception(self) -> str:
        """Calculate thermal perception value."""

        # https://en.wikipedia.org/wiki/Dew_point
        dew_point_c = self._calc_dew_point()
        if dew_point_c < 10:
            return "A bit dry for some"
        if dew_point_c < 13:
            return "Very comfortable"
        if dew_point_c < 16:
            return "Comfortable"
        if dew_point_c < 18:
            return "OK for most"
        if dew_point_c < 21:
            return "Somewhat uncomfortable"
        if dew_point_c < 24:
            return "Very humid, quite uncomfortable"
        if dew_point_c < 26:
            return "Extremely uncomfortable"
        return "Severely high"

    def _calc_heat_index(self) -> float:
        """Calculate the heat index."""

        temp_f = util.temperature.celsius_to_fahrenheit(self._temp_c)
        humidity = self._humidity

        # http://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml
        heat_index_f = 0.5 * (
            temp_f + 61.0 + ((temp_f - 68.0) * 1.2) + (humidity * 0.094)
        )

        if heat_index_f > 79:
            heat_index_f = -42.379 + 2.04901523 * temp_f
            heat_index_f += 10.14333127 * humidity
            heat_index_f += -0.22475541 * temp_f * humidity
            heat_index_f += -0.00683783 * pow(temp_f, 2)
            heat_index_f += -0.05481717 * pow(humidity, 2)
            heat_index_f += 0.00122874 * pow(temp_f, 2) * humidity
            heat_index_f += 0.00085282 * temp_f * pow(humidity, 2)
            heat_index_f += -0.00000199 * pow(temp_f, 2) * pow(humidity, 2)

        if humidity < 13 and temp_f >= 80 and temp_f <= 112:
            heat_index_f = heat_index_f - ((13 - humidity) * 0.25) * math.sqrt(
                (17 - abs(temp_f - 95)) * 0.05882
            )
        elif humidity > 85 and temp_f >= 80 and temp_f <= 87:
            heat_index_f += ((humidity - 85) * 0.1) * ((87 - temp_f) * 0.2)

        return util.temperature.fahrenheit_to_celsius(heat_index_f)

    def _calc_absolute_humidity(self) -> float:
        """Calculate absolute humidity."""

        temp_c = self._temp_c
        humidity = self._humidity

        # https://carnotcycle.wordpress.com/2012/08/04/how-to-convert-relative-humidity-to-absolute-humidity/
        abs_humidity = 6.112
        abs_humidity *= math.exp((17.67 * temp_c) / (243.5 + temp_c))
        abs_humidity *= humidity
        abs_humidity *= 2.1674
        abs_humidity /= temp_c + 273.15

        return abs_humidity

    def _calc_simmer_index(self) -> float:
        """Calculate summer simmer index."""

        temp_f = util.temperature.celsius_to_fahrenheit(self._temp_c)
        humidity = self._humidity

        # https://www.vcalc.com/wiki/rklarsen/Summer+Simmer+Index
        if temp_f < 70:
            simmer_index_f = temp_f
        else:
            simmer_index_f = (
                1.98 * (temp_f - (0.55 - (0.0055 * humidity)) * (temp_f - 58.0)) - 56.83
            )

        return util.temperature.fahrenheit_to_celsius(simmer_index_f)

    def _calc_simmer_danger(self) -> str | None:
        """Calculate summer simmer index danger."""

        simmer_index_c = self._calc_simmer_index()
        simmer_index_f = util.temperature.celsius_to_fahrenheit(simmer_index_c)
        if simmer_index_f < 70:
            return None
        if simmer_index_f < 77:
            return "Slightly cool"
        if simmer_index_f < 83:
            return "Comfortable"
        if simmer_index_f < 91:
            return "Slightly warm"
        if simmer_index_f < 100:
            return "Increasing discomfort"
        if simmer_index_f < 112:
            return "Extremely warm"
        if simmer_index_f < 125:
            return "Danger of heatstroke"
        if simmer_index_f < 150:
            return "Extreme danger of heatstroke"
        return "Circulatory collapse imminent"

    @property
    def should_poll(self) -> bool:
        """Return the polling state."""
        return False

    @property
    def name(self) -> str:
        """Return the name."""
        return self._name

    @property
    def device_class(self) -> str:
        """Return the device class of the sensor."""
        return self._device_class

    @property
    def native_unit_of_measurement(self) -> str:
        """Return the unit of measurement."""
        return self._unit_of_measurement

    @property
    def native_value(self) -> Any:
        """Return the state of the entity."""
        value = self._state
        if self._device_class == DEVICE_CLASS_TEMPERATURE:
            value = util.temperature.convert(
                value, TEMP_CELSIUS, self.hass.config.units.temperature_unit
            )
        return round(value, 2) if isinstance(value, float) else value

    @property
    def available(self) -> bool:
        """Return the availability of this sensor."""
        return self._available

    @property
    def extra_state_attributes(self) -> dict:
        """Return the state attributes."""
        temp = util.temperature.convert(
            self._temp_c, TEMP_CELSIUS, self.hass.config.units.temperature_unit
        )
        return {
            ATTR_TEMPERATURE: round(temp, 2),
            ATTR_HUMIDITY: round(self._humidity, 2),
        }
