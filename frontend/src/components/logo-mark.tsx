import { cn } from "@/lib/utils";

// Charter Bold's lowercase phi, drawn as a shape so it looks the same on every
// device (Charter is also the reading font). Same path as public/favicon.svg.
const PHI =
  "M14.47 20.7Q12.62 20.7 11.24 20.28Q9.87 19.85 8.95 19.04Q8.03 18.24 7.57 17.06Q7.11 15.87 7.11 14.36Q7.11 12.78 7.62 11.59Q8.13 10.4 9.03 9.6Q9.93 8.81 11.15 8.4Q12.38 8 13.82 8V9.31Q12.96 9.4 12.34 9.73Q11.73 10.07 11.33 10.69Q10.94 11.31 10.75 12.21Q10.56 13.12 10.56 14.36Q10.56 15.67 10.78 16.59Q11 17.51 11.48 18.11Q11.95 18.71 12.69 19.02Q13.43 19.33 14.47 19.4V12.36Q14.47 11.39 14.59 10.59Q14.71 9.79 15.12 9.21Q15.52 8.64 16.29 8.32Q17.05 8 18.33 8Q19.75 8 20.95 8.42Q22.16 8.83 23.03 9.63Q23.91 10.44 24.4 11.62Q24.89 12.81 24.89 14.36Q24.89 15.87 24.43 17.06Q23.97 18.24 23.05 19.04Q22.12 19.85 20.74 20.28Q19.37 20.7 17.53 20.7V26H14.47ZM17.53 19.4Q18.64 19.4 19.4 19.03Q20.15 18.65 20.6 17.99Q21.06 17.32 21.25 16.39Q21.44 15.47 21.44 14.36Q21.44 11.96 20.68 10.63Q19.93 9.31 18.5 9.31Q18.2 9.31 18.01 9.4Q17.82 9.49 17.72 9.7Q17.62 9.91 17.57 10.27Q17.53 10.63 17.53 11.16Z";

// The PhiloSearch mark: a lowercase serif phi on the brand blue.
export function LogoMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" aria-hidden="true" className={cn("w-7 h-7 shrink-0", className)}>
      <rect width="32" height="32" rx="6" fill="#1d4ed8" />
      <path fill="#ffffff" d={PHI} />
    </svg>
  );
}
