const PAGE_SIZE = 25;

const API_BASE_URL =
  window.location.hostname === 'localhost' ||
  window.location.hostname === '127.0.0.1'
    ? ''
    : 'https://customer-churn-ltv-engine-vqo6.onrender.com';

let schema;
let scoredRecords = [];
let currentPage = 0;


function apiUrl(path) {
  return `${API_BASE_URL}${path}`;
}


const elements = {
  activeModel: document.querySelector('#active-model-label'),
  status: document.querySelector('#service-status'),
  model: document.querySelector('#model-select'),
  horizon: document.querySelector('#horizon-input'),
  numericFields: document.querySelector('#numeric-fields'),
  categoricalFields: document.querySelector('#categorical-fields'),
  form: document.querySelector('#customer-form'),
  formMessage: document.querySelector('#form-message'),
  customerId: document.querySelector('#customer-id'),
  file: document.querySelector('#csv-file'),
  notice: document.querySelector('#result-notice'),
  table: document.querySelector('#table-wrap'),
  tableBody: document.querySelector('#results-body'),
  tableFootnote: document.querySelector('#table-footnote'),
  pageLabel: document.querySelector('#page-label'),
  previousPage: document.querySelector('#previous-page'),
  nextPage: document.querySelector('#next-page'),
  search: document.querySelector('#result-search'),
  count: document.querySelector('#metric-count'),
  high: document.querySelector('#metric-high'),
  revenue: document.querySelector('#metric-revenue'),
  metricHorizon: document.querySelector('#metric-horizon'),
};


const money = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  maximumFractionDigits: 0,
});


function displayModel(name) {
  return name
    .replaceAll('_', ' ')
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}


function setMessage(message, isError = false) {
  elements.formMessage.textContent = message;
  elements.formMessage.classList.toggle('is-error', isError);
}


function makeField(field) {
  const wrapper = document.createElement('label');
  wrapper.className = 'field-control';

  const caption = document.createElement('span');
  caption.textContent = field.name.replaceAll(
    /([a-z])([A-Z])/g,
    '$1 $2',
  );

  wrapper.append(caption);

  let control;

  if (field.kind === 'category') {
    control = document.createElement('select');

    for (const value of field.options) {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = value;
      control.append(option);
    }

    control.value = field.default;
  } else {
    control = document.createElement('input');
    control.type = 'number';

    control.step =
      field.name === 'SeniorCitizen' ||
      field.name === 'tenure'
        ? '1'
        : '0.01';

    control.min =
      field.name === 'SeniorCitizen'
        ? '0'
        : '0';

    if (field.name === 'SeniorCitizen') {
      control.max = '1';
    }

    control.value = field.default;
    control.required = true;
  }

  control.name = field.name;
  control.setAttribute('aria-label', field.name);

  wrapper.append(control);

  return wrapper;
}


function populateForm(fields) {
  for (const field of fields) {
    const parent =
      field.kind === 'number'
        ? elements.numericFields
        : elements.categoricalFields;

    parent.append(makeField(field));
  }
}


function updateModelLabel() {
  const selected = elements.model.value;
  elements.activeModel.textContent = displayModel(selected);
}


function customerFromForm() {
  const customer = {};

  for (const control of elements.form.querySelectorAll('[name]')) {
    customer[control.name] =
      control.type === 'number'
        ? Number(control.value)
        : control.value;
  }

  const customerId = elements.customerId.value.trim();

  if (customerId) {
    customer.customerID = customerId;
  }

  return customer;
}


async function parseResponse(response) {
  const payload = await response.json();

  if (!response.ok) {
    throw new Error(
      payload.detail ||
      'The request could not be completed.',
    );
  }

  return payload;
}


async function scoreProfile(customer) {
  const response = await fetch(
    apiUrl('/api/score'),
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        customers: [customer],
        model_name: elements.model.value,
        horizon_months: Number(
          elements.horizon.value,
        ),
      }),
    },
  );

  return parseResponse(response);
}


async function scoreCsv(file) {
  const query = new URLSearchParams({
    model_name: elements.model.value,
    horizon_months: elements.horizon.value,
  });

  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(
    `${apiUrl('/api/score-csv')}?${query}`,
    {
      method: 'POST',
      body: formData,
    },
  );

  return parseResponse(response);
}


function visibleRecords() {
  const query =
    elements.search.value.trim().toLowerCase();

  if (!query) {
    return scoredRecords;
  }

  return scoredRecords.filter(
    (record) =>
      String(record.customerID ?? '')
        .toLowerCase()
        .includes(query) ||
      record.risk_category
        .toLowerCase()
        .includes(query),
  );
}


function updateSummary(payload) {
  elements.count.textContent =
    payload.count.toLocaleString();

  elements.high.textContent =
    payload.high_risk_count.toLocaleString();

  elements.revenue.textContent =
    money.format(
      payload.total_revenue_at_risk,
    );

  elements.metricHorizon.textContent =
    `Across ${elements.horizon.value} months`;
}


function renderResults(payload) {
  scoredRecords = [...payload.records].sort(
    (left, right) =>
      right.churn_probability -
      left.churn_probability,
  );

  currentPage = 0;

  updateSummary(payload);

  elements.notice.hidden = true;
  elements.table.hidden = false;

  renderPage();
}


function renderPage() {
  const records = visibleRecords();

  const pageCount = Math.max(
    1,
    Math.ceil(records.length / PAGE_SIZE),
  );

  currentPage = Math.min(
    currentPage,
    pageCount - 1,
  );

  const first =
    currentPage * PAGE_SIZE;

  const page = records.slice(
    first,
    first + PAGE_SIZE,
  );

  elements.tableBody.replaceChildren();

  for (const [offset, record] of page.entries()) {
    const row = document.createElement('tr');

    const customer =
      record.customerID ||
      `Customer ${first + offset + 1}`;

    const cells = [
      customer,

      `${(
        record.churn_probability * 100
      ).toFixed(1)}%`,

      record.risk_category,

      money.format(
        record.monthly_charges,
      ),

      money.format(
        record.revenue_at_risk,
      ),

      money.format(
        record.estimated_ltv ?? 0,
      ),
    ];

    cells.forEach((value, index) => {
      const cell = document.createElement('td');

      if (index === 2) {
        const badge =
          document.createElement('span');

        badge.className =
          `risk-pill risk-${record.risk_category.toLowerCase()}`;

        badge.textContent = value;

        cell.append(badge);
      } else {
        cell.textContent = value;
      }

      row.append(cell);
    });

    elements.tableBody.append(row);
  }

  elements.tableFootnote.textContent =
    records.length
      ? `Sorted by churn score · ${records.length.toLocaleString()} result${records.length === 1 ? '' : 's'}`
      : 'No customers match this filter.';

  elements.pageLabel.textContent =
    records.length
      ? `${first + 1}–${Math.min(
          first + PAGE_SIZE,
          records.length,
        )} of ${records.length.toLocaleString()}`
      : '0 results';

  elements.previousPage.disabled =
    currentPage === 0;

  elements.nextPage.disabled =
    currentPage >= pageCount - 1;
}


async function initialize() {
  try {
    const [
      schemaResponse,
      healthResponse,
    ] = await Promise.all([
      fetch(apiUrl('/api/schema')),
      fetch(apiUrl('/health')),
    ]);

    schema =
      await parseResponse(schemaResponse);

    const health =
      await parseResponse(healthResponse);

    populateForm(schema.fields);

    elements.horizon.value =
      schema.default_horizon_months;

    for (const name of schema.models) {
      const option =
        document.createElement('option');

      option.value = name;
      option.textContent =
        displayModel(name);

      elements.model.append(option);
    }

    elements.model.value =
      schema.default_model;

    updateModelLabel();

    elements.status.classList.toggle(
      'is-ready',
      health.model_artifacts_ready,
    );

    elements.status.classList.toggle(
      'is-error',
      !health.model_artifacts_ready,
    );

    elements.status.lastChild.textContent =
      health.model_artifacts_ready
        ? 'Model ready'
        : 'Training required';

    if (!health.model_artifacts_ready) {
      setMessage(
        'Run the training command to create model files.',
        true,
      );
    }
  } catch (error) {
    elements.status.classList.add(
      'is-error',
    );

    elements.status.lastChild.textContent =
      'API unavailable';

    elements.activeModel.textContent =
      'Unavailable';

    setMessage(
      error.message,
      true,
    );
  }
}


elements.form.addEventListener(
  'submit',
  async (event) => {
    event.preventDefault();

    const button =
      elements.form.querySelector(
        'button[type="submit"]',
      );

    button.disabled = true;

    setMessage(
      'Scoring customer…',
    );

    try {
      const payload =
        await scoreProfile(
          customerFromForm(),
        );

      renderResults(payload);

      setMessage(
        'Customer scored.',
      );
    } catch (error) {
      setMessage(
        error.message,
        true,
      );
    } finally {
      button.disabled = false;
    }
  },
);


elements.file.addEventListener(
  'change',
  async () => {
    const [file] =
      elements.file.files;

    if (!file) return;

    setMessage(
      `Scoring ${file.name}…`,
    );

    try {
      const payload =
        await scoreCsv(file);

      renderResults(payload);

      setMessage(
        `${payload.count.toLocaleString()} customers scored.`,
      );
    } catch (error) {
      setMessage(
        error.message,
        true,
      );
    } finally {
      elements.file.value = '';
    }
  },
);


elements.model.addEventListener(
  'change',
  updateModelLabel,
);


elements.horizon.addEventListener(
  'change',
  () => {
    if (scoredRecords.length) {
      elements.metricHorizon.textContent =
        `Latest scores use ${scoredRecords[0].horizon_months} months`;
    }
  },
);


elements.search.addEventListener(
  'input',
  () => {
    currentPage = 0;
    renderPage();
  },
);


elements.previousPage.addEventListener(
  'click',
  () => {
    currentPage -= 1;
    renderPage();
  },
);


elements.nextPage.addEventListener(
  'click',
  () => {
    currentPage += 1;
    renderPage();
  },
);


initialize();