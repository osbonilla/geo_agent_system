/**
 * lib/markdown.ts
 * ───────────────
 * Markdown mínimo y seguro para las respuestas de los agentes:
 * **negrita**, *cursiva*, `código`, listas con - o 1., tablas | a | b |,
 * títulos con # y saltos de línea.
 *
 * Primero se escapa TODO el HTML (las respuestas pueden traer texto de
 * ArcGIS escrito por terceros), después se aplica el formato. Nunca se
 * inyecta HTML que venga del modelo.
 */
function escapar(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function enLinea(s: string): string {
  return s
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[^*])\*([^*\s][^*]*)\*/g, '$1<em>$2</em>');
}

export function markdown(texto: string): string {
  const lineas = escapar(texto ?? '').split(/\r?\n/);
  const out: string[] = [];
  let lista: 'ul' | 'ol' | null = null;

  const cerrar = () => {
    if (lista) out.push(`</${lista}>`);
    lista = null;
  };

  const celdas = (l: string) =>
    l.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map((c) => c.trim());
  const esSeparador = (l: string) => /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(l);

  for (let i = 0; i < lineas.length; i++) {
    const l = lineas[i];

    // Tabla: fila de encabezado seguida de |---|---|
    if (l.includes('|') && i + 1 < lineas.length && esSeparador(lineas[i + 1])) {
      cerrar();
      const cab = celdas(l);
      const filas: string[][] = [];
      i += 2;
      while (i < lineas.length && lineas[i].includes('|') && lineas[i].trim()) {
        filas.push(celdas(lineas[i]));
        i++;
      }
      i--;
      out.push(
        '<table><thead><tr>' + cab.map((c) => `<th>${enLinea(c)}</th>`).join('') + '</tr></thead><tbody>' +
          filas.map((f) => '<tr>' + cab.map((_, j) => `<td>${enLinea(f[j] ?? '')}</td>`).join('') + '</tr>').join('') +
          '</tbody></table>',
      );
      continue;
    }

    const ul = l.match(/^\s*[-*•]\s+(.*)$/);
    const ol = l.match(/^\s*\d+[.)]\s+(.*)$/);
    if (ul || ol) {
      const tipo = ul ? 'ul' : 'ol';
      if (lista !== tipo) {
        cerrar();
        out.push(`<${tipo}>`);
        lista = tipo;
      }
      out.push(`<li>${enLinea((ul ?? ol)![1])}</li>`);
    } else {
      cerrar();
      const h = l.match(/^#{1,4}\s+(.*)$/);
      if (h) out.push(`<p><strong>${enLinea(h[1])}</strong></p>`);
      else if (l.trim()) out.push(`<p>${enLinea(l)}</p>`);
    }
  }
  cerrar();
  return out.join('');
}